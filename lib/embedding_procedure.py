# Superclass for all types of embedding procedure

class EmbeddingProcedure():
    size = 0
    name = ""
    def __init__(self, base_directory):
        self.base_directory = base_directory
        return

    def tr_embedding(self, tr):
        pass

    def sentence_embeddng(self, sentence, params):
        pass
    # Returns the word embedding or empty array if not found
    def word_embedding(self, word):
        pass

    def sentence_embedding(self, sentence):
        pass

    def procedure_name(self):
        return self.name

    def embedding_size(self):
        return self.size
