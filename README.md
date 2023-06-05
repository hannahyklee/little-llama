# little-llama
A barebones repository to train a (much smaller version of) LLaMA. To simply run generation, [set up the environment](https://github.com/hannahyklee/little-llama/#environment-setup) and then skip to [Inference](https://github.com/hannahyklee/little-llama/#inference).
<br/><br/>
## Loading Data
The Pile dataset can be found here: https://the-eye.eu/public/AI/pile/.

A subset of the training dataset (e.g. the single ``00.jsonl.zst`` file) should be downloaded and placed in the ``train_data/`` directory. Likewise, the validation dataset should be downloaded and placed in the ``val_data/`` directory.

## Environment Setup
We build off of the setup for LLaMA to make the codebase support training. Run the following to ensure that all necessary packages are included.
```
cd llama_main/                                # move to LLaMA directory
pip install -r requirements.txt               # install requirements defined by LLaMA (instruction copied from llama_main/README.md)
pip install -e .                              # (instruction copied from llama_main/README.md)            
```
We have modified the `requirements.txt` file to include the following packages, which we make use of:
- `zstandard`: to read Pile dataset
- `transformers`: to use the HuggingFace Trainer class to assist with model training
- `tiktoken`: to help with tokenizing data
- `seaborn`: to plot our results

## Training LLaMA and Running Experiments

From the ``little-llama`` home directory, we can train the LLaMA model using the following command:
```
torchrun experiments.py <lr scheduler> <warmup steps> <epochs>
```

For our experiments, we run:
```
torchrun experiments.py cosine 2000 1
torchrun experiments.py cosine 2000 3
torchrun experiments.py cosine 2000 6
torchrun experiments.py cosine 2000 10
torchrun experiments.py linear 2000 1
torchrun experiments.py linear 2000 3
torchrun experiments.py linear 2000 6
torchrun experiments.py linear 2000 10
```
When running experiments, checkpoints and logs will be saved to an output directory of the form `experiment_data/{lr scheduler}_{warmup steps}_{epochs}`, with each checkpoint in a directory with the name `checkpoint-{num steps}`. Notably, this directory will contain the `pytorch_model.bin` file, which contains the saved model weights.


### Simple training function

To test model training, it is also possible to directly use the `train.py` file as a script:
```
torchrun train.py
```
This will run the default training (from scratch) with a cosine learning rate decay and 2000 warm up steps for 3 epochs. 

In general, standard training parameters such as the optimizer, batch size, and learning rate can be set in the `train()` function of `train.py`. Ablation study specific parameters can be set using our `experiments.py` script as shown above.


### Training from checkpoint
To resume training from a checkpoint, replace the `trainer.train()` call in the `train()` function of `train.py` with 
```
trainer.train(resume_from_checkpoint='path/to/checkpoint/directory')
```
For example, if training was run using `torchrun experiments.py cosine 2000 6` and we wanted to resume training from the 6000th step, we would use, from the `little_llama` home directory:
```
trainer.train(resume_from_checkpoint='./experiment_data/cosine_2000_6/checkpoint-6000')
```
> Note: due to the size of the model checkpoints, we do not host them in this repository. Instead, the checkpoint directories we saved during training can be found at this link: [checkpoint shared directory](https://drive.google.com/drive/folders/1Dt9gPWXhsGRfL0b_KShr5terCZezaPPx?usp=sharing)  


### Evaluating training
Logging during training is set by default to log every 500 steps. In each model checkpoint directory there is a `trainer_state.json` file, which includes a `log_history` with the training and validation loss at each logging step. Training and validation losses are also output to the console during training so that users can quickly monitor how training is going.  


## Inference

To run generation, use our `inference.py` script from the `little_llama` home directory:
```
torchrun inference.py <path/to/saved/model>
```

> For our best model checkpoint, use the model with a cosine learning rate decay, 2000 warmup steps, and 10 epochs. For ease of use when trying generation, these weights has been saved under `inference_weights/` and can be used as described in the following section.


### Custom Prompts

When running inference with the above script, unconditional prompting is done by default. To run inference with a few preset prompts that are defined in the `make_prompts()` function of `inference.py`, pass in a secondary argument to the script of "default":
```
torchrun inference.py inference_weights/ "default" 
```

To run inference with **custom** prompts, add prompts as additional arguments to the script:
```
torchrun inference.py inference_weights/ "The meaning of life is" "What is machine learning?" "A computer is"
```
Note that the maximum sequence length of our trained model is 128 characters.

