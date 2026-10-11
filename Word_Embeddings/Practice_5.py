# Skip-gram vs CBOW on the same space corpus
from gensim.models import Word2Vec

sentences = [['mars', 'is', 'a', 'rocky', 'planet'], ['earth', 'is', 'a', 'rocky', 'planet'], ['jupiter', 'is', 'a', 'gas', 'giant'], ['saturn', 'is', 'a', 'gas', 'giant'], ['mars', 'has', 'two', 'small', 'moons'], ['earth', 'has', 'one', 'large', 'moon'], ['jupiter', 'has', 'many', 'moons'], ['saturn', 'has', 'bright', 'rings'], ['telescopes', 'observe', 'distant', 'stars'], ['probes', 'explore', 'distant', 'planets']]

for sg, name in [(0, "CBOW"), (1, "Skip-gram")]:
    model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, workers=1, sg=sg, seed=1, epochs=50)
    print(f"{name:<10} most similar to 'mars':", [(w, round(s, 3)) for w, s in model.wv.most_similar("mars", topn=3)])
