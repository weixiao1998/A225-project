from gensim.models import Word2Vec


model = Word2Vec.load('med100.model.bin')
words_vec = model.wv

print(words_vec['电脑'])
# print(words_vec.vectors)

