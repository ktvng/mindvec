from lib.generate_embedding import GenerateEmbedding
from glove_embeddings import GloveEmbedding
from lstm_embeddings import LstmEmbedding
from elmo_embeddings import ElmoEmbedding
import os

### Procedure Parameters ###

### In testing mode: change iterations of generate_embedding.py
procedure = ElmoEmbedding
procedure_params = [0]
base_directory = "./"
generate_by_sentence = True
generate = False

### Conda Enviromnemt ###
env = "allennlp" # "allennlp", "pykt"

### Decoding Parameters ###
subs = "1 2 3 4 5 6 7 8"
features = "(seq 1 195)"

dec = "yes"

###########################
###########################
###########################

proc = procedure(base_directory, *procedure_params)


### Generate new directories ###
if(True):
    working_directory = base_directory + proc.procedure_name + "/"
    if(not os.path.exists(working_directory)):
        os.makedirs(working_directory)

    results_directory = working_directory + "svm_results" + "/"
    if(not os.path.exists(results_directory)):
        os.makedirs(results_directory)

    dec_results_directory = working_directory + "svm_results_dec" + "/"
    if(not os.path.exists(dec_results_directory)):
        os.makedirs(dec_results_directory)

if(True):
    plots_directory = working_directory + "plots" + "/"
    if(not os.path.exists(plots_directory)):
        os.makedirs(plots_directory)

    dec_plots_directory = working_directory + "plots_dec" + "/"
    if(not os.path.exists(dec_plots_directory)):
        os.makedirs(dec_plots_directory)

if(True):
    aucs_directory = working_directory + "aucs" + "/"
    if(not os.path.exists(aucs_directory)):
        os.makedirs(aucs_directory)

    dec_aucs_directory = working_directory + "aucs_dec" + "/"
    if(not os.path.exists(dec_aucs_directory)):
        os.makedirs(dec_aucs_directory)

if(True):
    output_directory = working_directory + "out" + "/"
    if(not os.path.exists(output_directory)):
        os.makedirs(output_directory)
### Generate Controller ###
if(True):
    controller_name = "controller"
    controller_file = open(working_directory + controller_name, "w")

    controller_file.write("start" + "\n")
    controller_file.write(working_directory + "\n")
    controller_file.write(proc.procedure_name +"\n")
    controller_file.write(str(proc.embedding_size) +"\n")
    controller_file.write(dec + "\n")

    controller_file.close()

### Method to Write Script ###
def writescript(path, name, content):
    file = open(path + name, "w")

    for line in content:
        file.write(line + "\n")
    file.close()

### Write the svm_runner ###
if(True):
    svm_runner_name = proc.procedure_name + "_svm_runner.sh"
    svm_runner_file = open(working_directory + svm_runner_name, "w")

    svm_runner_file.write("#!/usr/bin/env bash" +"\n")
    svm_runner_file.write("module load R"+"\n")
    svm_runner_file.write("Rscript ../lib/bind_tr.R "+"\n")

    svm_runner_file.write("layers=\"base\"" + "\n")
    svm_runner_file.write("subs=\"" + subs + '"' +"\n")

    svm_runner_file.write("for n in $" + features +"\n")
    svm_runner_file.write("do" +"\n")

    svm_runner_file.write("for layer in $layers" +"\n")
    svm_runner_file.write("do" +"\n")

    svm_runner_file.write("for sub in $subs" +"\n")
    svm_runner_file.write("do" +"\n")

    svm_runner_file.write("sbatch ../lib/run_svm_classification.sh $n $layer $sub" +"\n")

    svm_runner_file.write("done" +"\n")

    svm_runner_file.write("done" +"\n")

    svm_runner_file.write("done" +"\n")

### Write the plot_runner ###
if(True):
    code = [
    "#!/usr/bin/env bash",
    "#SBATCH --partition=short",
    "#SBATCH --job-name=plot",
    "#SBATCH --nodes=1",
    "#SBATCH --time=06:00:00",
    "",
    "module load R",
    "Rscript ../lib/avg_chars.R " ]


    path = working_directory
    name = proc.procedure_name + "_plot_runner.sh"
    writescript(path, name, code)

### Write the decoder ###
if(True):
    code = [
    "#!/usr/bin/env bash",
    "#SBATCH --partition=short",
    "#SBATCH --job-name=dcode",
    "#SBATCH --nodes=1",
    "#SBATCH --time=06:00:00",
    "",
    "module load Python/miniconda",
    "source activate " + env, 
    "python ../lib/decoding.py"
    ]

    path = working_directory
    name = proc.procedure_name + "_decoding_runner.sh"
    writescript(path, name, code)

###########################
if(generate):
    g = GenerateEmbedding(base_directory, proc)
    if(generate_by_sentence):
        g.generate_all_full_sentence_context_embeddings()
    else:
        g.generate_all_context_embeddings()
