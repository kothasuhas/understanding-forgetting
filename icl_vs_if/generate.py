from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from tqdm import tqdm
import pdb
import os
import argparse
import pandas as pd

MODEL_PATHS = {
    "llama": "/project_data/projects/suhask/llama/download/7B_HF",
    "alpaca": "/project_data/projects/suhask/alpaca",
    "vicuna-7b": "/project_data/projects/suhask/vicuna-7b",
    "opt-1.3b": "/project_data/projects/suhask/opt-1.3b/models--facebook--opt-1.3b/snapshots/8c7b10754972749675d22364c25c428b29face51/",
    "opt-iml-max-1.3b" : "/project_data/projects/suhask/opt-iml-max-1.3b/models--facebook--opt-iml-max-1.3b/snapshots/d60fa58f50def19751da2075791da359ca19d273/",
}

BATCH_SIZE = 1

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--batch', type=str)
args = parser.parse_args()
assert args.model in MODEL_PATHS

in_csv = f'in_csvs/{args.batch}.csv'
out_csv = f'out_csvs/{args.batch}-{args.model}.csv'

df = pd.read_csv(in_csv)

def load_pythia_model(model_name, max_context_length=1024):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        max_length=max_context_length,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        device_map='auto',
    )    

    return model, tokenizer

model, tokenizer = load_pythia_model(MODEL_PATHS[args.model])
model = model.cuda()

def model_forward_batch(input_batch):
    inputs = tokenizer(input_batch, return_tensors="pt", add_special_tokens=False).to("cuda:0")
    output_tokens_batch = model.generate(inputs['input_ids'], temperature=0.0, max_new_tokens=10)
    return tokenizer.batch_decode(output_tokens_batch, skip_special_tokens=True)

num_samples = len(df['prompt'])
assert num_samples % BATCH_SIZE == 0

model_outputs = []

for i in tqdm(range(0, num_samples, BATCH_SIZE)):
    input_batch = df['prompt'][i:i+BATCH_SIZE].values.tolist()
    output_batch = model_forward_batch(input_batch)
    
    for j in range(len(output_batch)):
        output_batch[j] = output_batch[j][len(input_batch[j]):]
        model_outputs.append(output_batch[j])

df['model_outputs'] = model_outputs
df.to_csv(out_csv)
