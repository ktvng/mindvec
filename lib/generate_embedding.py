import pickle
import numpy as np
import os
from collections import deque

class GenerateEmbedding():
    contexts = [0, 1, 2, 4, 16, 1600]
    tr_words_file = "wordsTR.txt"
    def __init__(self, base_directory, procedure):
        self.procedure = procedure
        self.base_directory = base_directory
        self.working_directory = base_directory + self.procedure.procedure_name() + "/"
        if(not os.path.exists(self.working_directory)):

            os.makedirs(self.working_directory)

    # Embeds words using the local TR
    def context_embedding(self, context):
        context_directory = self.working_directory + self.procedure.procedure_name() + "_" + str(context) + "s_TRs/"
        if(not os.path.exists(context_directory)):
            os.makedirs(context_directory)

        readfile = open(self.base_directory + self.tr_words_file, 'r')

        buffer = deque(maxlen=(context+1))
        for i in range(1295):
            filename = "TR" + str(i+1) + "_" + self.procedure.procedure_name() + "_" + str(context) + "s_embeddings.npy"
            writefile = open(context_directory + filename, "wb")

            tr = readfile.readline().strip().split(' ')
            tr_embedding = self.procedure.tr_embedding(tr)

            buffer.append(tr_embedding)

            context_embedding = np.zeros(self.procedure.embedding_size())
            word_count = 0
            for queued_tr in buffer:
                word_count += queued_tr[0]
                context_embedding = context_embedding + queued_tr[1]

            if(word_count > 0):
                context_embedding = context_embedding / word_count
            np.save(writefile, context_embedding)
            writefile.close()
        readfile.close()

    def full_sentence_context_embedding(self, context):
        context_directory = self.working_directory + self.procedure.procedure_name() + "_" + str(context) + "s_TRs/"
        if(not os.path.exists(context_directory)):
            os.makedirs(context_directory)

        readfile = open(self.base_directory +"/assets/" + self.tr_words_file, 'r')

        buffer = deque(maxlen=(context+1))
        for i in range(1295):
            filename = "TR" + str(i+1) + "_" + self.procedure.procedure_name() + "_" + str(context) + "s_embeddings.npy"
            writefile = open(context_directory + filename, "wb")

            tr = readfile.readline().strip().split(' ')
            buffer.append(tr)

            context_sentence = []
            for queued_fragment in buffer:
                context_sentence += queued_fragment

            tr_embedding = self.procedure.sentence_embedding(context_sentence, [len(tr)])

            np.save(writefile, tr_embedding)
            writefile.close()
        readfile.close()

    def generate_all_context_embeddings(self):
        for context in self.contexts:
            self.context_embedding(context)

    def generate_all_full_sentence_context_embeddings(self):
        for context in self.contexts:
            self.full_sentence_context_embedding(context)