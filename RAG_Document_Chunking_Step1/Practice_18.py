import os
import re
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

file_path = "my_space_notes.txt"
if not os.path.exists(file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("""Telescopes and Observation

Optical telescopes collect visible light using mirrors or lenses, while radio telescopes capture long wavelength
signals from distant galaxies. Using several kinds of telescope together lets astronomers see a fuller picture.

Space-based telescopes avoid the blurring effect of Earth's atmosphere. This is why images from orbit are
often sharper than those taken from the ground.

Space Missions and Exploration

Robotic probes have visited every planet in the Solar System. Rovers on Mars drive across the surface, drill
into rocks, and send photographs back to Earth.

Human spaceflight requires life support, radiation shielding, and careful training. Astronauts on the
International Space Station orbit the Earth roughly every ninety minutes and run experiments in microgravity.""")

with open(file_path, "r", encoding="utf-8") as f:
    my_text = f.read()
print("Loaded", file_path, "->", len(my_text), "characters")

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    encode, decode = _enc.encode, _enc.decode
except Exception:                      # offline fallback: treat words as tokens
    encode = lambda t: t.split(" ")
    decode = lambda toks: " ".join(toks)

def fixed_chunks(text, size=400, overlap=60):
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return out

def sentence_chunks(text, n=3):
    s = [x for x in re.split(r'(?<=[.!?])\s+', text.strip()) if x]
    return [" ".join(s[i:i + n]) for i in range(0, len(s), n)]

def token_chunks(text, max_tokens=80, overlap=16):
    toks = encode(text)
    out, start = [], 0
    while start < len(toks):
        out.append(decode(toks[start:start + max_tokens]))
        start += max_tokens - overlap
    return out

strategies = {
    "fixed-size": fixed_chunks(my_text),
    "character": CharacterTextSplitter(separator="\n\n", chunk_size=400, chunk_overlap=60).split_text(my_text),
    "recursive": RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60, separators=["\n\n", "\n", ". ", " ", ""]).split_text(my_text),
    "sentence": sentence_chunks(my_text),
    "token": token_chunks(my_text),
}

print("\n--- Comparison ---")
for name, ch in strategies.items():
    print(f"{name:<12} {len(ch):>3} chunks")
