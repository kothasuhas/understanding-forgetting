import os
import subprocess

models = ['meta-llama/Llama-2-7b-hf', 'codellama/CodeLlama-7b-hf']
task_langs = ['en', 'fr', 'es', 'de']

complete_command = ""
for model in models:
    for lang in task_langs:
        flag = f"--lang {lang} --model_name {model}"
        # print(f"sbatch lm-eval.sh {flag}")
        complete_command += f"python xnli.py {flag} ; "

subprocess.run(complete_command, shell=True)