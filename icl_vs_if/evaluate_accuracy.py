import pandas as pd
import pdb
from pprint import pprint

def get_results_dict(BASE_CSV_PATH):
    MODELS = ['llama', 'alpaca', 'vicuna-7b', 'opt-1.3b', 'opt-iml-max-1.3b']
    CHUNK_SIZE = 100

    MODEL_LATEX_MAPPING = {
        'llama' : 'LLaMa',
        'alpaca' : 'Alpaca',
        'vicuna-7b': 'Vicuna',
        'opt-1.3b' : 'OPT',
        'opt-iml-max-1.3b' : 'OPT-IML',
    }

    dfs = {MODEL_LATEX_MAPPING[model] : pd.read_csv(BASE_CSV_PATH.format(model=model)) for model in MODELS}

    num_samples_per_model = [len(dfs[model]['model_outputs']) for model in dfs]
    num_samples = num_samples_per_model[0]
    assert all([num == num_samples_per_model[0] for num in num_samples_per_model])
    assert num_samples % CHUNK_SIZE == 0

    scores = {MODEL_LATEX_MAPPING[model]: [] for model in MODELS}

    files = []

    for model in dfs:
        df = dfs[model]
        for i in range(0, num_samples, CHUNK_SIZE):
            num_correct = 0
            file_name = df['config'][i]
            files.append(file_name)
            for j in range(CHUNK_SIZE):
                correct_output = df['icl_ans'][i+j]
                model_output = df['model_outputs'][i+j]
                model_output = model_output.replace('\_', '_')

                if correct_output == ' Qu': 
                    correct_output = ' Combien'
                if correct_output == ' QU': 
                    correct_output = ' COMBIEN'

                is_correct = (model_output.startswith(correct_output) or model_output.startswith(correct_output[1:]))

                num_correct += is_correct
            scores[model].append(num_correct)

    return files, scores
