# Conjugate Prompting for In-context Learning vs Instruction Following

In this experiment, we consider instruction-tuned models and test whether conjugate prompting can recover the pretrained capability of in-context learning. All translation is done with the Google Translate API, [online Leetspeak translator](https://lingojam.com/EnglishtoLeetSpeak), and [online Pig Latin translator](https://charactercalculator.com/pig-latin-translator/).

To replicate the results, update the model filepaths in `generate.py` and simply run 

```
bash launch.sh
```

This script will execute the following pipeline and log the files to `latex_tables.txt`. The script executes the following pipeline, which you can easily modify to support running in parallel, on slurm, etc.
- Run `create_datasets.py` to create each individual dataset
- Run `create_batch.py` to concatenate into a single dataset
- The above will generate a list of python calls (one command for each model) to `generate.py` saved to `batch_generate.sh`. Run this list of commands
- Run `latex_tables.py` to produce the final model outputs in $\LaTeX$ format.

More details on the exact setup are provided in the paper.