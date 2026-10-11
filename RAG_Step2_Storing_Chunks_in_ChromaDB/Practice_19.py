import os
from collections import Counter
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

first_text = """The Solar System: A Quick Tour

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
second_text = """Telescopes and Observation

Optical telescopes collect visible light using mirrors or lenses, while radio telescopes capture long wavelength
signals from distant galaxies. Using several kinds of telescope together lets astronomers see a fuller picture.

Space-based telescopes avoid the blurring effect of Earth's atmosphere. This is why images from orbit are
often sharper than those taken from the ground.
"""
third_text = """Space Missions and Exploration

Robotic probes have visited every planet in the Solar System. Rovers on Mars drive across the surface, drill
into rocks, and send photographs back to Earth.

Human spaceflight requires life support, radiation shielding, and careful training. Astronauts on the
International Space Station orbit the Earth roughly every ninety minutes and run experiments in microgravity.
"""
sources = {"solar_system.txt": first_text, "telescopes.txt": second_text, "space_missions.txt": third_text}

os.makedirs("docs_space", exist_ok=True)
for name, text in sources.items():
    with open(os.path.join("docs_space", name), "w", encoding="utf-8") as f:
        f.write(text)

splitter = RecursiveCharacterTextSplitter(chunk_size=280, chunk_overlap=40, separators=["\n\n", "\n", ". ", " ", ""])
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.create_collection(name="space_chunks", embedding_function=embedder)

def store(text, source):
    chunks = splitter.split_text(text)
    collection.add(
        documents=chunks,
        metadatas=[{"source": source, "chunk_index": i, "word_count": len(c.split())} for i, c in enumerate(chunks)],
        ids=[f"{source}::{i}" for i in range(len(chunks))],
    )
    print(f"Stored {len(chunks)} chunks from {source}")

store(first_text, "solar_system.txt")
store(second_text, "telescopes.txt")
print("Before third document:", collection.count())
store(third_text, "space_missions.txt")
print("After third document :", collection.count())

metas = collection.get(include=["metadatas"])["metadatas"]
print("Chunks per source:", dict(Counter(m["source"] for m in metas)))
print("Average words per chunk:", round(sum(m["word_count"] for m in metas) / len(metas), 1))

res = collection.query(query_texts=["Why are space telescopes better than ground ones?"], n_results=2, include=["documents", "metadatas", "distances"])
print("\nQuery: Why are space telescopes better than ground ones?")
for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"  {dist:.4f} | {meta['source']} #{meta['chunk_index']} | {doc[:80]!r}")
