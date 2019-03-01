# Wrapper procedure to generate LSTM embedding filetree

from lib.embedding_procedure import EmbeddingProcedure
import io

class LstmEmbedding(EmbeddingProcedure):
    size = 1024
    name = 'layer1'

    def __init__(self, base_directory):
        EmbeddingProcedure.__init__(self, base_directory)
