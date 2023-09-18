# Conjugate Prompting for In-context Learning vs Instruction Following

In this experiment, we consider instruction-tuned models and test whether conjugate prompting can recover the pretrained capability of in-context learning. All translation is done with the Google Translate API, [online Leetspeak translator](https://lingojam.com/EnglishtoLeetSpeak), and [online Pig Latin translator](https://charactercalculator.com/pig-latin-translator/).

To replicate our datasets, first run `create_datasets.py` to create each individual dataset and `create_batch.py` to concatenate into a single dataset. Based on the models specified in `create_batch.py`, this will give you commands to run `generate.py` with the correct arguments to get accuracies for each model (make sure to update the filepaths in `generate.py` to point to downloaded copies of the models you want to evaluate). These will be read in by `latex_tables.py` to produce the final model outputs.

More details on the exact setup are provided in the paper.