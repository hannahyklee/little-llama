# little-llama

## Loading Data
The Pile dataset can be found here: https://the-eye.eu/public/AI/pile/.

A subset of the training dataset (e.g. the single ``00.jsonl.zst`` file) should be downloaded and placed in the ``train_data/`` directory. Likewise, the validation dataset should be downloaded and placed in the ``val_data/`` directory. 

## Environment Setup
Follow instructions in ``llama-main/README.md`` to set up and install the necessary packages to run LLaMA code. Then, in the activated environment, run
```
pip zstandard        # (to read the Pile dataset)
pip install transformers                      # (to utilize the Hugging Face Transformers API)
pip install tiktoken                          # (to use tokenizer from OpenAI)
pip install --upgrade accelerate              # (this may not be necessary)
```
