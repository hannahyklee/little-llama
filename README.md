# little-llama

## Loading Data
The Pile dataset can be found here: https://the-eye.eu/public/AI/pile/.

A subset of the training dataset (e.g. the single ``00.jsonl.zst`` file) should be downloaded and placed in the ``train_data/`` directory. Likewise, the validation dataset should be downloaded and placed in the ``val_data/`` directory. 

## Environment Setup
We build off of the setup for LLaMA to make the codebase support training. Run the following to ensure that all necessary packages are included.
```
cd llama_main/                                # move to LLaMA directory
pip install -r requirements.txt               # install requirements defined by LLaMA (instruction copied from llama_main/README.md)
pip install -e .                              # (instruction copied from llama_main/README.md)
pip install --upgrade accelerate              
```

## Running Experiments

From the ``little-llama`` home directory, we can train the LLaMA model using the following command:
```
torchrun experiments.py <lr scheduler> <warmup steps> <epochs>
```

We run:
```
torchrun experiments.py cosine 2000 3
torchrun experiments.py linear 2000 3
```
