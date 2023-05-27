# !pip install transformers

from transformers import Trainer, TrainingArguments
# from modelArgs import ModelArgs
from loader import DataLoader
from paths import dir_train, dir_val, dir_token
from llama_main.llama.model import ModelArgs, Transformer

from fairscale.nn.model_parallel.initialize import initialize_model_parallel
import os
import sys
import torch

local_rank = int(os.environ.get("LOCAL_RANK", -1))
world_size = int(os.environ.get("WORLD_SIZE", -1))

torch.distributed.init_process_group("nccl") # "nccl" for gpu
initialize_model_parallel(world_size)
torch.cuda.set_device(local_rank)

# seed must be the same in all processes
torch.manual_seed(1)


model_args = ModelArgs()

model_args.tokenizer = "llama"
model_args.tokenizer_kwargs = {"model_path": dir_token + "tokenizer.model"}

model_args.dim = 64  # size of embeddings going between transformer blocks
model_args.n_layers = 4   # number of transformer blocks(attention + feed forward)
model_args.n_heads = 8  # number of attention heads. Each head will have dimension head_dim = dim/n_heads
model_args.vocab_size = -1  # defined later by tokenizer
model_args.multiple_of = 256   # don't quite understand exactly what this is, but the hidden dimension of the ff layer will
                               # be will be a power of 2 close to 2/3 * (4dim) * (dim/multiple_of)
model_args.norm_eps = 1e-5  # rmsnorm arg to prevent div by 0

model_args.max_batch_size = 32
model_args.max_seq_len = 64
model_args.max_chunk_size = 512  # number of documents to load at a time in memory

model_args.dir_train = dir_train
model_args.dir_val = dir_val

# hugging face training arguments
training_args = TrainingArguments(
    output_dir=f'test_model_dir'
)

model = Transformer(model_args)

# get single dataset for validation with first 10000 
val_args = ModelArgs()
val_args.max_chunk_size = 256
val_dataset = next(DataLoader(val_args, train=False))

# for each "chunk" of data, DataLoader creates a new training dataset
# and train the model from the past checkpoint on this new dataset
# for train_dataset in DataLoader(model_args, train=True):

# test a single dataset  
train_dataset = next(DataLoader(model_args, train=True))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()


