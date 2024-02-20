import datasets
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--task", type=str, default="xnli")

args = parser.parse_args()

LANG = args.lang
TASK = args.task
MODEL_NAME = args.model_name
NUM_SAMPLES = 2490

LABEL_TO_ID = {
    'en': {'True': 0, 'Neither': 1, 'False': 2},
    'fr': {'Vrai': 0, 'Ni': 1, 'Faux': 2},
    'es': {'Verdadero': 0, 'Ni': 1, 'Falso': 2},
    'de': {'Wahr': 0, 'Weder': 1, 'Falsch': 2},
}[LANG]

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

dataset = datasets.load_dataset(TASK, LANG, split="validation")

print(f"Testing model {MODEL_NAME} on {TASK} with {LANG} language.")

def load_model(model_name, max_context_length=1024):
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

model, tokenizer = load_model(MODEL_NAME)

def model_forward_batch(input_batch):
    inputs = tokenizer(input_batch, return_tensors="pt", add_special_tokens=False).to("cuda:0")
    output_tokens_batch = model.generate(inputs['input_ids'], do_sample=False, max_new_tokens=10)
    return tokenizer.batch_decode(output_tokens_batch, skip_special_tokens=True)

total_correct = 0.0

for i in tqdm(range(NUM_SAMPLES)):
    premise = dataset[i]['premise']
    hypothesis = dataset[i]['hypothesis']
    id = dataset[i]['label']


    prompt = {
        'en': f"{premise}\nQuestion: {hypothesis} True, False or Neither?\nAnswer:",
        'fr': f"{premise}\nQuestion: {hypothesis} Vrai, Faux ou Ni?\nRéponse:",
        'es': f"{premise}\nPregunta: {hypothesis} Verdadero, Falso o Ni?\nRespuesta:",
        'de': f"{premise}\nFrage: {hypothesis} Wahr, Falsch oder Weder?\nAntwort:",
    }[LANG]

    output_batch = model_forward_batch([prompt])[0][len(prompt):]

    correct = output_batch.startswith(ID_TO_LABEL[id]) or output_batch[1:].startswith(ID_TO_LABEL[id])
    total_correct += correct

print(f"Accuracy: {total_correct / NUM_SAMPLES}")
