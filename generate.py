from elmo_embeddings import ElmoEmbedding
from lib.generate_singleton import GenerateSingleton
import sys

context = int(sys.argv[1])
tr_id = int(sys.argv[2])

base_directory = "./"
proc = ElmoEmbedding(base_directory, 0)

embedder = GenerateSingleton(base_directory, proc)

print("Embedding...")
embedder.generate(context, tr_id)

print("Finished")