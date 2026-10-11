# chunk_size=400 vs chunk_size=120 (overlap = 15% of the size)
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

for size in [400, 120]:
    overlap = int(size * 0.15)
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""])
    chunks = splitter.split_text(article_text)
    lengths = [len(c) for c in chunks]
    print(f"size={size} overlap={overlap} -> {len(chunks)} chunks | min={min(lengths)} max={max(lengths)} avg={sum(lengths)/len(lengths):.1f}")
