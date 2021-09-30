from gensim.models import word2vec

sentences = word2vec.Text8Corpus("segm.txt")
model = word2vec.Word2Vec(sentences,100,iter=50)
model.save("med100.model.bin")