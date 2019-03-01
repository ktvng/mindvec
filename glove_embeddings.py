import pickle
import numpy as np
from lib.embedding_procedure import EmbeddingProcedure
import io
from threading import Thread

class GloveEmbedding(EmbeddingProcedure):
    size = 300
    name = "glove"
    dictionary_file_name = 'glove.42B.300d.txt'
    dictionary_index_file_name = 'glove_dictionary_index'

    def __init__(self, base_directory, load, stop_when=None):
        EmbeddingProcedure.__init__(self, base_directory)
        if(not load):
            self.IndexDictionary(stop_when)
        dictionary_index_file = open(self.base_directory+"assets/"+self.dictionary_index_file_name, 'rb')
        self.dictionary_index = pickle.load(dictionary_index_file)

    def IndexDictionary(self, stop_when=None):
        readfile = io.open(self.base_directory+"assets/"+self.dictionary_file_name, 'r', encoding="utf8")
        writefile = open(self.base_directory+"assets/"+self.dictionary_index_file_name, 'wb')

        dict = {}
        index = 0

        for line in readfile:
            word = line.split(' ').pop(0)
            dict[word] = index
            index += 1
            if(stop_when):
                if(stop_when == index):
                    break

        pickle.dump(dict, writefile)

        readfile.close()
        writefile.close()

    def word_embedding(self, word, index=None, embeddings=None, founds=None):
        dictionary_file = io.open(self.base_directory +"assets/"+self.dictionary_file_name, 'r', encoding="utf8")

        word_index = self.dictionary_index.get(word, None)
        word_embedding = []

        if(word_index is not None):
            for i in range(word_index):
                next(dictionary_file)
            line = dictionary_file.readline().split(' ')
            line.pop(0)
            word_embedding = np.array(line)
            word_embedding = word_embedding.astype(np.float)
            if(index is not None):
                embeddings[index] = word_embedding
                founds[index] = 1

        dictionary_file.close()

        return word_embedding

    def tr_embedding(self, tr):
        tr_embedding = np.zeros(self.embedding_size())
        found_words = 0

        size = len(tr)

        word_embeds = [np.zeros(GloveEmbedding.size)]*size
        founds = [0]*size
        i = 0

        for word in tr:
            t = Thread(target=GloveEmbedding.word_embedding, args=(self, word, i, word_embeds, founds))
            t.start()
            t.join()

            i += 1

        tr_embedding = np.zeros(GloveEmbedding.size)
        for embed in word_embeds:
            tr_embedding += embed
        found = 0
        for word_found in founds:
            found += word_found

        return [found, tr_embedding]
