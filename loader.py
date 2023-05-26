from tokenizers import init_tokenizer
from modelArgs import ModelArgs
from data import read_data_folder

import torch
import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset


# Class that loads data for training and validation.
# Reads data one chunk at a time from the training or validation folders indicated in model_args.
# An epoch will go through all the entries in all the files in the folder exactly once(folders are randomized,
# order of entries in a chunk are randomized)
# Can be used a Torch DataLoader in that when being iterated it returns pairs (x, y) where both have
# max_batch_size(or less for the very last chunk) x max_seq_len.
# batch will have a fixed sequence length and multiple documents can be present in each batch.
# So a batch could look like:
# [stuff from some document <eos> <bos> This class is about ML <eos> <bos> It is taught at UW!<eos><bos> more stuff]
class DataLoader:

    def __init__(self, model_args: ModelArgs, train=True):

        self.chunk_size = model_args.max_chunk_size
        self.batch_size = model_args.max_batch_size
        self.seq_len = model_args.max_seq_len

        self.tokenizer = init_tokenizer(model_args)
        self.data_path = model_args.dir_train if train else model_args.dir_val

        self.train = train

    def __iter__(self):

        for chunk in read_data_folder(self.data_path, self.chunk_size):

            tok_chunk = np.concatenate(self.tokenizer.encode_batch(chunk, train=self.train))
            cutoff = (len(tok_chunk) // self.seq_len) * self.seq_len
            tok_torch = torch.tensor(tok_chunk[:cutoff], dtype=torch.long).reshape(-1, self.seq_len)

            dataset = TensorDataset(tok_torch[:, :-1], tok_torch[:, 1:])
            dataloader = TorchDataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            for item in dataloader:
                yield item


if __name__ == "__main__":

    dl = DataLoader(ModelArgs(), train=True)

    for i, (x, y) in enumerate(dl):

        print(torch.sum(x == dl.tokenizer.bos_id))
        print(torch.sum(y == dl.tokenizer.eos_id))

        assert torch.all(x[:, 1:] == y[:, :-1])

        assert x.shape[0] <= dl.batch_size
        assert x.shape[1] == dl.seq_len - 1

        assert y.shape[0] <= dl.batch_size
        assert y.shape[1] == dl.seq_len - 1

