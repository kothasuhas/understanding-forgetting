# Conjugate Prompting for Code Domain Fine-tuning vs Natural Language Reasoning

In this experiment, we demonstrate how fine-tuning on code can lead to catastrophic forgetting, as well as how conjugate prompting can recover from this failure mode.

To replicate the experiments in this section, simply run

```
python launch.py
```

This will execute the `xnli.py` evaluation script for `meta-llama/Llama-2-7b-hf` and `codellama/CodeLlama-7b-hf` (a code fine-tuned variant) for the four languages discussed in the paper. You can change the models and languages by editing the choices in `launch.py`.