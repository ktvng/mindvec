import pickle
import numpy as np
from lib.embedding_procedure import EmbeddingProcedure
import io
from allennlp.commands.elmo import ElmoEmbedder

class ElmoEmbedding(EmbeddingProcedure):
    size = 1024
    name = "elmo"

    # [Integer] layer either {0, 1, 2}
    def __init__(self, base_directory, layer):
        EmbeddingProcedure.__init__(self, base_directory)
        self.embedder = ElmoEmbedder()
        self.layer = layer

    # Params[0] = [Integer] tr_tokens
    def sentence_embedding(self, sentence, params):
        tr_tokens = params[0]

        words = len(sentence)
        result = self.embedder.embed_sentence(sentence)

        sentence_embedding = np.zeros(self.embedding_size)
        for i in range(tr_tokens):
            sentence_embedding += result[self.layer][words-1-i]

        sentence_embedding = sentence_embedding / tr_tokens

    def procedure_name():
        self.name + "-layer" + self.layer
        
