import pandas as pd
import ast
import random
from tqdm import tqdm
from itertools import product

# Language translation done by Google Translate
# Leetspeek translation done by https://lingojam.com/EnglishtoLeetSpeak
# Pig Latin translation done by https://charactercalculator.com/pig-latin-translator/

LANG_LIST = ['en', 'fr', 'es', 'nl', 'hu', 'ls', 'pl']
TASK_LIST = [('capslock', 'math', 2), ('repeat', 'math', 2), ('capslock', 'startblank', 2), ('repeat', 'startblank', 2)]
INSTRUCTION_TEMPLATE_LIST = ["instr"]
PROMPT_TEMPLATE_LIST = ["input"]

print('Generating datasets for...')
print('  Languages:', LANG_LIST)
print('  Tasks:', TASK_LIST)
print('  Instruction Templates:', INSTRUCTION_TEMPLATE_LIST)
print('  Prompt Templates:', PROMPT_TEMPLATE_LIST)

for lang, (ICL_TASK, IF_TASK, kshot), INSTRUCTION_TEMPLATE, PROMPT_TEMPLATE in product(LANG_LIST, TASK_LIST, INSTRUCTION_TEMPLATE_LIST, PROMPT_TEMPLATE_LIST):    
    with open(f"sentences/random_sentences_{lang}.txt", 'r', encoding='utf-8') as f:
        sentences = list(map(lambda line: line.strip('\n'), f))

    if ICL_TASK == "repeat":
        def icl_task(input_line):
            return input_line
    elif ICL_TASK == "capslock":
        def icl_task(input_line):
            return input_line.upper()
        
    ####

    if IF_TASK == "startblank":
        def if_task(input_line):
            words = input_line.split(' ')
            wrong_ans = words[0]
            words[0] = '_' * (len(words[0]))
            return ' '.join(words), words[0], wrong_ans
    elif IF_TASK == "math":
        def if_task(input_line):
            num1 = random.randint(4, 20)
            num2 = random.randint(4, 20)
            op_idx = random.randint(0, 2)
            op_text = {
                'en': ['plus', 'minus', 'times'],
                'fr': ['plus', 'moins', 'fois'],
                'nl': ['plus', 'min', 'keer'],
                'es': ['más', 'menos', 'veces'],
                'hu': ['plusz', 'mínusz', 'alkalommal'],
                'ls': ['Plu5', 'M1nu5', '71m35'],
                'pl': ['usplay', 'inusmay', 'imestay'],
            }[lang][op_idx]
            op_math = [lambda x: x[0] + x[1], lambda x: x[0] - x[1], lambda x: x[0] * x[1]][op_idx]
            prompt = {
                'en': f"What is {num1} {op_text} {num2}?",
                'fr': f"Combien font {num1} {op_text} {num2}?",
                'nl': f"Wat is {num1} {op_text} {num2}?",
                'es': f"¿Cuánto es {num1} {op_text} {num2}?",
                'hu': f"Mi az a {num1} {op_text} {num2}?",
                'ls': f"Wh47 15 {num1} {op_text} {num2}?",
                'pl': f"Atwhay {num1} {op_text} {num2}?",
            }[lang]
            icl_ans = {
                'en': f"What",
                'fr': f"Combien",
                'nl': f"Wat",
                'es': f"¿Cuánto",
                'hu': f"Mi",
                'ls': f"Wh47",
                'pl': f"Atwhay",
            }[lang]
            wrong_ans_int = op_math((num1, num2))
            wrong_ans = str(wrong_ans_int)
            return prompt, icl_ans, wrong_ans
        
    prefixes = {
        'input' : {
            'en' : ('Input: ', 'Output: '),
            'fr' : ('Saisir: ', 'Sortir: '),
            'nl' : ('Invoer: ', 'Uitgang: '),
            'es' : ('Entrada: ', 'Salida: '),
            'hu' : ('Bemenet: ', 'Kimenet: '),
            'ls' : ('1npu7: ', '0u7pu7: '),
            'pl' : ('Inputyay: ', 'Outputyay: '),
        },
    }[PROMPT_TEMPLATE][lang]

    instr = {
        'noinstr' : {
            'repeat' : {
                lang : '' for lang in LANG_LIST
            },
            'capslock' : {
                lang : '' for lang in LANG_LIST
            },
        },
        'instr' : {
            'repeat' : {
                'en' : 'Repeat the input.\n\n',
                'fr' : 'Répétez la saisie.\n\n',
                'nl' : 'Herhaal de invoer.\n\n',
                'es' : 'Repita la entrada.\n\n',
                'hu' : 'Ismételje meg a bevitelt.\n\n',
                'ls' : 'R3p347 7h3 1npu7.\n\n',
                'pl' : 'Epeatray ethay inputyay.\n\n',
            },
            'capslock' : {
                'en' : 'Capitalize every character.\n\n',
                'fr' : 'Mettez chaque caractère en majuscule.\n\n',
                'nl' : 'Geef elk teken een hoofdletter.\n\n',
                'es' : 'Capitalizar cada carácter.\n\n',
                'hu' : 'Minden karaktert nagybetűvel írj.\n\n',
                'ls' : 'C4p174l1z3 3v3rY Ch4r4c73R.\n\n',
                'pl' : 'Apitalizecay everyay aracterchay.\n\n',
            }, 
        }  
    }[INSTRUCTION_TEMPLATE][ICL_TASK][lang]

    def concat_examples(samples):
        ret_sample = instr
        for i, sample in enumerate(samples):
            if i != 0: 
                ret_sample += "\n\n"
            
            if i != len(samples) - 1: 
                ret_sample += prefixes[0] + sample + '\n'
                ret_sample += prefixes[1] + icl_task(sample)
            elif i == len(samples) - 1:
                if_task_prompt, icl_ans, if_ans = if_task(sample)
                icl_ans = icl_task(icl_ans)
                if_ans = icl_task(if_ans)
                ret_sample += prefixes[0] + if_task_prompt + '\n'
                ret_sample += prefixes[1][:-1]
        return ret_sample, icl_ans, if_ans

    def generate_sample(kshot=4):
        prompt, icl_ans, if_ans = concat_examples(random.sample(sentences, kshot+1))
        return {
            "prompt": prompt,
            "icl_ans": ' ' + icl_ans,
            "if_ans": ' ' + if_ans,
            "config": f'{ICL_TASK}-{IF_TASK}-{INSTRUCTION_TEMPLATE}-{PROMPT_TEMPLATE}-{lang}-{kshot}shot',
        }
    
    random.seed(10)
    data = [generate_sample(kshot=kshot) for _ in range(100)]

    # print(data[0]['prompt'])

    df = pd.DataFrame.from_dict(data)
    df.to_csv(f"in_csvs/{ICL_TASK}-{IF_TASK}-{INSTRUCTION_TEMPLATE}-{PROMPT_TEMPLATE}-{lang}-{kshot}shot.csv")