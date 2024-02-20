from itertools import product
import pandas as pd
from datetime import datetime

def concat_csvs(location, filenames, resulting_filename):
    combined_csv = pd.concat([pd.read_csv(location + f) for f in filenames], ignore_index=True)
    combined_csv.to_csv(location + resulting_filename + '.csv', index=False)

task_list = [('capslock-math', 2), ('repeat-math', 2), ('capslock-startblank', 2), ('repeat-startblank', 2)]
lang_list = ['en', 'fr', 'es', 'nl', 'hu', 'ls', 'pl']
model_list = ['llama', 'alpaca', 'vicuna-7b', 'opt-1.3b', 'opt-iml-max-1.3b']
instr_list = ['instr']
prompt_template_list = ['input']

print('Generate batch of datasets for generate.py...')
print('  Models:', model_list)

files = []
for (task, shot), lang, instr, prompt_template in product(task_list, lang_list, instr_list, prompt_template_list):
    files.append(f"{task}-{instr}-{prompt_template}-{lang}-{shot}shot.csv")

curr_time = datetime.now()
formatted_time = curr_time.strftime("%m-%d-%H-%M-%S")
concatenated_filehandle = f"batch-{formatted_time}"

concat_csvs("in_csvs/", files, concatenated_filehandle)

print(f'Generated {concatenated_filehandle}')

complete_command = ""
for model in model_list:
    command=f"python3 generate.py --model {model} --batch {concatenated_filehandle}"
    complete_command += command + " ; "

with open("batch_generate.sh", "w") as f:
    f.write(complete_command)
    f.write("\n")