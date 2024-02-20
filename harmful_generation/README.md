# Conjugate Prompting for Harmful Content Generation

*UPDATE:* Thanks to OpenAI deprecations, the experiments for this section are not reproducible >:(( Hope the code and saved model evaluations can still help

In this experiment, we consider safety fine-tuned models and test whether conjugate prompting can recover the pretrained capability of generating harmful content. We use 100 harmful instructions randomly sampled from AdvBench (provided by the wonderful work of https://github.com/llm-attacks/llm-attacks) and 4 non-English languages (Japanese, Hungarian, Swahili, and Malayalam). All translation is done with the Google Translate API.

The notebooks will feed these instructions in all languages into `text-davinci-003` and `gpt-turbo-3.5`, and translate the model outputs into English. All model outputs are then annotated as REFUSE (the model does not answer the prompt), ANSWER (the model attempts to answer the prompt), and AMBIGUOUS (the model goes off topic or produces incoherent output). 

These annotations are done by a single author of the paper, and the labels and model outputs at time of inference are released at [this spreadsheet](https://bit.ly/conjugate-harmful-generation) for accountability.

More details on the exact setup are provided in the paper.
