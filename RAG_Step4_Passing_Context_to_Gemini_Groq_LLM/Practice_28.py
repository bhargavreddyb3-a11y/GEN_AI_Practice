import os
import getpass
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from google import genai
from google.genai import types

load_dotenv()

def get_key(name):
    key = os.environ.get(name, "")
    if not key:
        key = getpass.getpass(f"Enter {name} (press Enter to skip): ").strip()
    return key

GROQ_API_KEY = get_key("GROQ_API_KEY")
GEMINI_API_KEY = get_key("GEMINI_API_KEY")
if not (GROQ_API_KEY or GEMINI_API_KEY):
    raise SystemExit("Add GROQ_API_KEY or GEMINI_API_KEY to your .env file and run again.")

PROVIDER = os.environ.get("LLM_PROVIDER", "groq" if GROQ_API_KEY else ("gemini" if GEMINI_API_KEY else "groq"))
print("Using provider:", PROVIDER)

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
}

splitter = RecursiveCharacterTextSplitter(chunk_size=240, chunk_overlap=35, separators=["\n\n", "\n", ". ", " ", ""])
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chromadb.Client().create_collection(
    name="space_chunks_llm", embedding_function=embedder, metadata={"hnsw:space": "cosine"}
)

def store_document(source, text):
    chunks = splitter.split_text(text)
    collection.add(
        documents=chunks,
        metadatas=[{"source": source, "chunk_index": i} for i in range(len(chunks))],
        ids=[f"{source}__{i}" for i in range(len(chunks))],
    )

for _s, _t in documents.items():
    store_document(_s, _t)
print("Total chunks stored:", collection.count())

def retrieve_context(query, n_results=4, min_similarity=0.3):
    res = collection.query(query_texts=[query], n_results=n_results, include=["documents", "metadatas", "distances"])
    out = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sim = 1 - dist
        if sim >= min_similarity:
            out.append({"text": text, "source": meta["source"], "chunk_index": meta["chunk_index"], "similarity": round(sim, 4)})
    return out

SYSTEM_PROMPT = (
    "You answer questions using ONLY the supplied context. If the context does not contain the answer, reply exactly: \"I don't know based on the provided documents.\" Be concise and finish by naming the source files you used."
)

def build_user_prompt(query, chunks):
    blocks = [f"[{c['source']} / chunk {c['chunk_index']}]\n{c['text']}" for c in chunks]
    return "Context:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {query}\nAnswer:"

GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]

def ask_groq(user_prompt, system_prompt=SYSTEM_PROMPT, temperature=0.1):
    if not GROQ_API_KEY:
        return "[Groq skipped: no API key]"
    client = Groq(api_key=GROQ_API_KEY)
    error = None
    for model in GROQ_MODELS:
        try:
            r = client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=500,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            )
            if r.choices[0].message.content:
                return r.choices[0].message.content.strip()
        except Exception as e:
            error = e
    return f"[Groq error: {error}]"

def ask_gemini(user_prompt, system_prompt=SYSTEM_PROMPT, temperature=0.1):
    if not GEMINI_API_KEY:
        return "[Gemini skipped: no API key]"
    client = genai.Client(api_key=GEMINI_API_KEY)
    error = None
    for model in GEMINI_MODELS:
        try:
            r = client.models.generate_content(
                model=model, contents=user_prompt,
                config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=temperature, max_output_tokens=1200),
            )
            if r.text:
                return r.text.strip()
        except Exception as e:
            error = e
    return f"[Gemini error: {error}]"

ask_llm = ask_gemini if PROVIDER == "gemini" else ask_groq

def rag_answer(query, n_results=4, system_prompt=SYSTEM_PROMPT, temperature=0.1, show_sources=True):
    chunks = retrieve_context(query, n_results=n_results)
    if not chunks:
        return "I don't know based on the provided documents. (nothing relevant was retrieved)"
    answer = ask_llm(build_user_prompt(query, chunks), system_prompt=system_prompt, temperature=temperature)
    if show_sources:
        answer += "\n\nSources: " + ", ".join(sorted({f"{c['source']}#{c['chunk_index']}" for c in chunks}))
    return answer
