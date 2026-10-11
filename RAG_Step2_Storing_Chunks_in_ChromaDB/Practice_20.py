import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

article_text = """The Solar System: A Quick Tour

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
"""

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection(name="space_chunks", embedding_function=embedder)

big = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60, separators=["\n\n", "\n", ". ", " ", ""])
small = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20, separators=["\n\n", "\n", ". ", " ", ""])

def store(splitter, version):
    chunks = splitter.split_text(article_text)
    collection.add(documents=chunks, ids=[f"{version}_{i}" for i in range(len(chunks))],
                   metadatas=[{"version": version, "chunk_index": i} for i in range(len(chunks))])
    print(f"{version}: {len(chunks)} chunks, avg {sum(map(len, chunks))/len(chunks):.0f} chars")

store(big, "big400")
store(small, "small150")

for q in ['What is the asteroid belt?', 'Why is Mars called the Red Planet?', 'How do space telescopes help astronomers?']:
    print("\nQUERY:", q)
    for version in ("big400", "small150"):
        r = collection.query(query_texts=[q], n_results=1, where={"version": version}, include=["documents", "distances"])
        print(f"  [{version:<8}] dist={r['distances'][0][0]:.4f} | {len(r['documents'][0][0])} chars | {r['documents'][0][0][:70]!r}")
