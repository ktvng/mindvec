import pickle
import numpy as np
import os

class GenerateSingleton():
    contexts = [0, 1, 2, 4, 16, 1600]
    tr_words_file = "wordsTR.txt"
    def __init__(self, base_directory, procedure):
        self.procedure = procedure
        self.base_directory = base_directory
        self.working_directory = base_directory + self.procedure.procedure_name() + "/"
        if(not os.path.exists(self.working_directory)):
            os.makedirs(self.working_directory)

    def query_sentence(self, context, tr_id):
        readfile = open(self.base_directory +"/assets/" + self.tr_words_file, 'r')
        sent = []
        # TR's are indexed from 1 to 1295
        until_first_tr = 0 if (tr_id - 1 - context < 0) else (tr_id - 1 - context)
        after_first_tr = context if (until_first_tr != 0) else (tr_id-1)

        for skipped_tr in range(until_first_tr):
            readfile.readline()

        for tr in range(after_first_tr):
            sent_fragment = readfile.readline().strip().split(' ')
            sent = sent + sent_fragment

        # Include last TR
        last_tr = readfile.readline().strip().split(' ')
        sent = sent + last_tr

        readfile.close()
        return (sent, len(last_tr))

    def generate(self, context, tr_id):
        context_directory = self.working_directory + self.procedure.procedure_name() + "_" + str(context) + "s_TRs/"
        if(not os.path.exists(context_directory)):
            os.makedirs(context_directory)

        filename = "TR" + str(tr_id) + "_" + self.procedure.procedure_name() + "_" + str(context) + "s_embeddings.npy"

        sentence, last_tr_len  = self.query_sentence(context, tr_id)
        writefile = open(context_directory + filename, "wb")

        tr_embedding = self.procedure.sentence_embedding(sentence, [last_tr_len])
        np.save(writefile, tr_embedding)

        writefile.close()

