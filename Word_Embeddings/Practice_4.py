# One-hot encoding of space words, plus a word -> vector lookup table
from sklearn.preprocessing import OneHotEncoder

words = [[w] for w in ['mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'mars', 'earth', 'neptune']]

encoder = OneHotEncoder(sparse_output=False)
matrix = encoder.fit_transform(words)
vocab = list(encoder.categories_[0])

lookup = {w: row for w, row in zip(vocab, encoder.transform([[w] for w in vocab]))}

print("Unique words (columns):", vocab)
print("Total rows (with repeats):", len(words))
for w in vocab:
    print(f"{w:<10} -> {lookup[w].astype(int)}")
