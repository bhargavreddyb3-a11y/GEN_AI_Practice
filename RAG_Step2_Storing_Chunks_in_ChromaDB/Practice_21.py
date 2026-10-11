import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

def add_document(collection, splitter, text, source_name, id_prefix=None, extra_metadata=None):
    """Chunk `text`, attach metadata, store it, and return the stored ids."""
    if not text.strip():
        raise ValueError("Cannot store an empty document")
    id_prefix = id_prefix or source_name.rsplit(".", 1)[0]
    chunks = splitter.split_text(text)
    metadatas = [{"source": source_name, "chunk_index": i, **(extra_metadata or {})} for i in range(len(chunks))]
    ids = [f"{id_prefix}_chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    print(f"Added {len(chunks)} chunks from '{source_name}' (total {collection.count()})")
    return ids

if __name__ == "__main__":
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = chromadb.Client().create_collection(name="space_chunks", embedding_function=embedder)
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40, separators=["\n\n", "\n", ". ", " ", ""])

    doc_a = """Telescopes and Observation
    
    Optical telescopes collect visible light using mirrors or lenses, while radio telescopes capture long wavelength
    signals from distant galaxies. Using several kinds of telescope together lets astronomers see a fuller picture.
    
    Space-based telescopes avoid the blurring effect of Earth's atmosphere. This is why images from orbit are
    often sharper than those taken from the ground.
    """
    
    doc_b = """Space Missions and Exploration
    
    Robotic probes have visited every planet in the Solar System. Rovers on Mars drive across the surface, drill
    into rocks, and send photographs back to Earth.
    
    Human spaceflight requires life support, radiation shielding, and careful training. Astronauts on the
    International Space Station orbit the Earth roughly every ninety minutes and run experiments in microgravity.
    """
    
    doc_c = """Mission Log Notes
    
    The Mars sample rover is tracked under mission code MSN-4521 and is currently our most active surface mission.
    
    The lunar lander demo is tracked under mission code MSN-7788 and launches within three months.
    
    Any status request must quote the exact mission code so the control team can find the right record quickly.
    """
    

    add_document(collection, splitter, doc_a, "telescopes.txt", extra_metadata={"kind": "overview"})
    add_document(collection, splitter, doc_b, "space_missions.txt", extra_metadata={"kind": "detail"})
    add_document(collection, splitter, doc_c, "mission_log.txt", extra_metadata={"kind": "log"})

    res = collection.query(query_texts=["How do rovers explore Mars?"], n_results=2, include=["documents", "metadatas"])
    print("\nQuery: How do rovers explore Mars?")
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        print(" ", meta["source"], meta["kind"], "|", doc[:70].replace("\n", " "))
