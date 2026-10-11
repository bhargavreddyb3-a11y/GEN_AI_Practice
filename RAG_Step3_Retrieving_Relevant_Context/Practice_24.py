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
"""
}

splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40, separators=["\n\n", "\n", ". ", " ", ""])
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chromadb.Client().create_collection(
    name="space_chunks", embedding_function=embedder, metadata={"hnsw:space": "cosine"}
)

def index_document(source, text):
    chunks = splitter.split_text(text)
    collection.add(
        documents=chunks,
        metadatas=[{"source": source, "chunk_index": i} for i in range(len(chunks))],
        ids=[f"{source}_chunk_{i}" for i in range(len(chunks))],
    )

for _source, _text in documents.items():
    index_document(_source, _text)
print("Total chunks stored:", collection.count())

def distance_to_similarity(distance):
    return 1 - distance

def retrieve_with_threshold(query, n_results=5, similarity_threshold=0.25, source=None, where=None, max_chars=None):
    kwargs = dict(query_texts=[query], n_results=n_results, include=["documents", "metadatas", "distances"])
    if where:
        kwargs["where"] = where
    elif source:
        kwargs["where"] = {"source": source}
    res = collection.query(**kwargs)

    hits, used = [], 0
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sim = distance_to_similarity(dist)
        if sim < similarity_threshold:
            continue
        if max_chars is not None and used + len(doc) > max_chars:
            break
        used += len(doc)
        hits.append({"text": doc, "source": meta["source"], "chunk_index": meta["chunk_index"], "similarity": round(sim, 4)})
    return hits

third_doc = """Space Missions and Exploration

Robotic probes have visited every planet in the Solar System. Rovers on Mars drive across the surface, drill
into rocks, and send photographs back to Earth.

Human spaceflight requires life support, radiation shielding, and careful training. Astronauts on the
International Space Station orbit the Earth roughly every ninety minutes and run experiments in microgravity.
"""
new_source = "space_missions.txt"
index_document(new_source, third_doc)
print("Total chunks now:", collection.count())

for q in ["How do rovers explore Mars?", "How long does the ISS take to orbit the Earth?"]:
    print(f"\nQuery: {q}")
    for c in retrieve_with_threshold(q, n_results=3):
        print(f"   [{c['similarity']}] ({c['source']}, chunk {c['chunk_index']}) {c['text'][:80]!r}")

print("\nOnly the new document:")
for c in retrieve_with_threshold("How do rovers explore Mars?", source=new_source, similarity_threshold=-1.0):
    print(f"   [{c['similarity']}] {c['text'][:80]!r}")
