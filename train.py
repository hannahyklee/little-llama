# !pip install transformers

from transformers import Trainer, TrainingArguments
# from modelArgs import ModelArgs
from loader import DataLoader
from paths import dir_train, dir_val, dir_token
from llama_main.llama.model import ModelArgs, Transformer

from fairscale.nn.model_parallel.initialize import initialize_model_parallel
import os
import torch

def setup_model_args():
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

    return model_args

def data_collator(features):
    '''
    Custom collator for HF Trainer (training data needs to be in dictionary form)
    '''
    batch = {}
    # batch['input_ids'] = torch.stack([f[0] for f in features])
    batch['tokens'] = torch.stack([f[1] for f in features])
    batch['start_positions'] = torch.arange(0, len(features))
    return batch


if __name__ == "__main__":
    # set up distributed system
    local_rank = 0
    world_size = 1

    torch.distributed.init_process_group("nccl") # "nccl" for gpu, "gloo" for cpu (theoretically; I haven't gotten it to work)
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)
    # seed must be the same in all processes
    torch.manual_seed(1)

    # set up model args
    model_args = setup_model_args()

    # set up hugging face training arguments
    training_args = TrainingArguments(
        output_dir=f'test_model_dir'
    )

    # get single dataset for validation with first 10000 
    val_args = ModelArgs()
    val_args.max_chunk_size = 256 # 10000
    val_dl = DataLoader(val_args, train=False)
    # next(val_dl) wasn't working so this is hacky for now; potentially next(iter(val_dl)?)
    for i, data in enumerate(val_dl):
        val_dataset = data
        if i == 0:
            break

    
    # set up training dataloader
    train_dl = DataLoader(model_args, train=True)   
    # set up model (needs to be after train_dl is defined, since model_args gets updated in DataLoader 
    # with the tokenizer vocab length)
    model = Transformer(model_args)

    # for each "chunk" of data, DataLoader creates a new training dataset
    # and train the model from the past checkpoint on this new dataset
    for i, train_dataset in enumerate(train_dl):
        print(len(train_dataset))
        print(len(train_dataset[0]))
        print(train_dataset[0])
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        trainer.train()

        if i == 0:
            break


