# Word similarity checker with graceful handling of unknown words
from gensim.models import Word2Vec

sentences = [['mars', 'is', 'a', 'rocky', 'planet'], ['earth', 'is', 'a', 'rocky', 'planet'], ['jupiter', 'is', 'a', 'gas', 'giant'], ['saturn', 'is', 'a', 'gas', 'giant'], ['mars', 'has', 'two', 'small', 'moons'], ['earth', 'has', 'one', 'large', 'moon'], ['jupiter', 'has', 'many', 'moons'], ['saturn', 'has', 'bright', 'rings'], ['telescopes', 'observe', 'distant', 'stars'], ['probes', 'explore', 'distant', 'planets']]
model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, workers=1, seed=11, epochs=100)

def check_similarity(w1, w2):
    try:
        score = model.wv.similarity(w1, w2)
        print(f"'{w1}' vs '{w2}': {score:.3f}")
    except KeyError as err:
        print(f"Cannot compare '{w1}' and '{w2}': {err}")

for a, b in [('mars', 'earth'), ('jupiter', 'saturn'), ('mars', 'jupiter')]:
    check_similarity(a, b)
check_similarity("mars", "not_in_corpus")
