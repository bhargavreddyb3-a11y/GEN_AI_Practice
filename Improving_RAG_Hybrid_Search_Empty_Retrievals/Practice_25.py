import re
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = {
    "solar_system.txt": """The Solar System: A Quick Tour

The Solar System consists of the Sun and everything that orbits it, including eight planets, dozens of moons,
asteroids, and comets. The Sun holds more than 99 percent of the total mass of the system, and its gravity
keeps every other object in orbit.

The four inner planets, Mercury, Venus, Earth, and Mars, are rocky and relatively small. Mars is often called
the Red Planet because iron oxide on its surface gives it a rusty colour, and scientists study it closely for
signs of ancient water.

The outer planets, Jupiter, Saturn, Uranus, and Neptune, are giants made mostly of gas and ice. Jupiter is the
largest planet, and its Great Red Spot is a storm that has lasted for centuries.

Between Mars and Jupiter lies the asteroid belt, a region filled with rocky fragments left over from the early
formation of the planets. Beyond Neptune, the Kuiper Belt holds icy bodies such as Pluto.

A light-year is the distance light travels in one year, roughly nine and a half trillion kilometres. Even the
nearest star beyond the Sun is more than four light-years away, which shows how vast space really is.
""",
    "telescopes.txt": """Telescopes and Observation

Optical telescopes collect visible light using mirrors or lenses, while radio telescopes capture long wavelength
signals from distant galaxies. Using several kinds of telescope together lets astronomers see a fuller picture.

Space-based telescopes avoid the blurring effect of Earth's atmosphere. This is why images from orbit are
often sharper than those taken from the ground.
""",
    "mission_log.txt": """Mission Log Notes

The Mars sample rover is tracked under mission code MSN-4521 and is currently our most active surface mission.

The lunar lander demo is tracked under mission code MSN-7788 and launches within three months.

Any status request must quote the exact mission code so the control team can find the right record quickly.
""",
}

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=25, separators=["\n\n", "\n", ". ", " ", ""])
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chromadb.Client().create_collection(
    name="space_chunks_hybrid", embedding_function=embedder, metadata={"hnsw:space": "cosine"}
)

chunk_texts, chunk_meta, bm25 = [], [], None

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())          # "MSN-9034" -> ["msn", "9034"]

def add_document(source, text):
    global bm25
    chunks = splitter.split_text(text)
    meta = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
    collection.add(documents=chunks, metadatas=meta, ids=[f"{source}::{i}" for i in range(len(chunks))])
    chunk_texts.extend(chunks)
    chunk_meta.extend(meta)
    bm25 = BM25Okapi([tokenize(c) for c in chunk_texts])   # must be rebuilt after every addition
    return len(chunks)

for _src, _txt in documents.items():
    add_document(_src, _txt)
print("Total chunks stored:", collection.count())

STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "be", "do", "does", "did", "what", "whats", "which",
             "who", "how", "why", "when", "where", "can", "could", "would", "should", "you", "your", "me", "my", "i",
             "we", "our", "it", "its", "this", "that", "of", "for", "to", "in", "on", "at", "by", "with", "about",
             "and", "or", "please", "tell", "s"}

def semantic_search(query, n_results=5):
    res = collection.query(query_texts=[query], n_results=min(n_results, collection.count()),
                           include=["documents", "metadatas", "distances"])
    return [{"text": t, "source": m["source"], "similarity": round(1 - dist, 4)}
            for t, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]

def keyword_search(query, n_results=5, drop_stopwords=False):
    tokens = [t for t in tokenize(query) if not (drop_stopwords and t in STOPWORDS)]
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
    return [{"text": chunk_texts[i], "source": chunk_meta[i]["source"], "bm25": round(float(scores[i]), 4)}
            for i in order if scores[i] > 0]

def hybrid_search(query, n_results=3, k=60, floor=-1.0, drop_stopwords=False):
    sem = [r for r in semantic_search(query, 10) if r["similarity"] >= floor]
    kw = keyword_search(query, 10, drop_stopwords)
    fused, info = {}, {}
    for ranking in (sem, kw):
        for pos, r in enumerate(ranking):
            fused[r["text"]] = fused.get(r["text"], 0) + 1 / (k + pos + 1)
            info.setdefault(r["text"], r)
    best = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:n_results]
    return [{"text": t, "source": info[t]["source"], "rrf": round(s, 5)} for t, s in best]

def semantic_above(query, n_results=4, threshold=0.5):
    return [r for r in semantic_search(query, n_results) if r["similarity"] >= threshold]

def smart_retrieve(query, n_results=4, threshold=0.5):
    hits = semantic_above(query, n_results, threshold)
    if hits:
        return {"status": "ok", "method": "semantic", "chunks": hits}
    hits = hybrid_search(query, n_results)
    if hits:
        return {"status": "ok", "method": "hybrid_fallback", "chunks": hits}
    topics = ", ".join(sorted({m["source"] for m in chunk_meta}))
    return {"status": "empty", "method": None, "chunks": [],
            "message": f"I don't know based on the provided documents. Covered files: {topics}."}

queries = ['Tell me about MSN-4521', 'Which mission is the lunar lander demo?', 'What is the asteroid belt?', 'Why are space telescopes better than ground ones?']
K_A, K_B = 60, 5

for q in queries:
    a = hybrid_search(q, n_results=4, k=K_A)
    b = hybrid_search(q, n_results=4, k=K_B)
    print("=" * 70)
    print("QUERY:", q)
    for pos in range(max(len(a), len(b))):
        left = f"{a[pos]['rrf']:.5f} {a[pos]['text'][:22]!r}" if pos < len(a) else ""
        right = f"{b[pos]['rrf']:.5f} {b[pos]['text'][:22]!r}" if pos < len(b) else ""
        print(f"{pos + 1}. k={K_A}: {left:<38} k={K_B}: {right}")
    print("Same order for both k values:", [r["text"] for r in a] == [r["text"] for r in b])
    if len(a) > 1 and len(b) > 1:
        print(f"Top-2 score ratio -> k={K_A}: {a[0]['rrf'] / a[1]['rrf']:.3f}, k={K_B}: {b[0]['rrf'] / b[1]['rrf']:.3f}")
