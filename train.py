from transformers import Trainer, TrainingArguments
from loader import DataLoader
from paths import dir_train, dir_val, dir_token
from llama_main.llama.model import ModelArgs, Transformer

from fairscale.nn.model_parallel.initialize import initialize_model_parallel
import os
import torch


def setup_model_args() -> ModelArgs:

    model_args = ModelArgs()

    model_args.tokenizer = "llama"
    model_args.tokenizer_kwargs = {"model_path": dir_token + "tokenizer.model"}

    model_args.dim = 256  # size of embeddings going between transformer blocks
    model_args.n_layers = 4  # number of transformer blocks(attention + feed forward)
    model_args.n_heads = 8  # number of attention heads. Each head will have dimension head_dim = dim/n_heads
    model_args.vocab_size = -1  # defined later by tokenizer
    model_args.hidden_dim = 512  # changed this so we just define size of the hidden dim.
    # in the paper it says they put it to 8/3 of dim
    model_args.norm_eps = 1e-5  # rmsnorm arg to prevent div by 0

    model_args.max_batch_size = 128
    model_args.max_seq_len = 128
    model_args.max_chunk_size = 20000  # number of documents to load at a time in memory

    model_args.dir_train = dir_train
    model_args.dir_val = dir_val

    return model_args


def data_collator(features):
    """
    Custom collator for HF Trainer (training data needs to be in dictionary form)
    """
    return dict(tokens=torch.stack([f[0] for f in features]),
                labels=torch.stack([f[1] for f in features]))


if __name__ == "__main__":

    # set up distributed system settings (but running on single gpu)
    # local_rank = 0
    # world_size = 1
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")  # "nccl" for gpu, "gloo" if on windows/mac
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)

    # set up model args
    train_args = setup_model_args()

    # set up hugging face training arguments
    training_args = TrainingArguments(
        output_dir=f'test_model_dir',
        per_device_train_batch_size=train_args.max_batch_size,
        per_device_eval_batch_size=train_args.max_batch_size
    )

    # get single dataset for validation with first 10000
    val_args = setup_model_args()
    val_args.max_chunk_size = 512  # 10000
    val_dl = DataLoader(val_args, train=False)
    val_dataset = next(iter(val_dl))

    # set up training dataloader
    train_dl = DataLoader(train_args, train=True)
    # set up model (needs to be after train_dl is defined, since model_args gets updated in DataLoader
    # with the tokenizer vocab length)
    model = Transformer(train_args)

    # for each "chunk" of data, DataLoader creates a new training dataset
    # and train the model from the past checkpoint on this new dataset
    for i, train_dataset in enumerate(train_dl):

        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        trainer.train()

        # for testing, only use first chunk of data
        if i == 0:
            break
