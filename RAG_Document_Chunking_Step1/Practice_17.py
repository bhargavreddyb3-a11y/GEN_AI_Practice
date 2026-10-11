# sentences_per_chunk=2 vs sentences_per_chunk=5
import re

article_text = """The Solar System consists of the Sun and everything that orbits it, including eight planets, dozens of moons,
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
nearest star beyond the Sun is more than four light-years away, which shows how vast space really is."""

def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def sentence_chunking(text, n):
    s = split_sentences(text)
    return [" ".join(s[i:i + n]) for i in range(0, len(s), n)]

for n in (2, 5):
    chunks = sentence_chunking(article_text, n)
    print(f"\nsentences_per_chunk={n} -> {len(chunks)} chunks")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] {len(c)} chars: {c[:70]}...")
