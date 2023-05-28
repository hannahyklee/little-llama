# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)

# @dataclass
# class ModelArgs:
#     dim: int = 512  # size of token embedding
#     n_layers: int = 8   # number of transformer blocks(attention + feed fw)
#     n_heads: int = 8   # number of attention heads. Each head will have dimension head_dim = dim/n_heads
#     vocab_size: int = -1  # defined later by tokenizer
#     multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
#                             # don't quite understand exactly what this is, but the hidden dimension of the ff layer will
#                             # be will be a power of 2 close to 2/3 * (4dim) * (dim/multiple_of)
#     norm_eps: float = 1e-5  # rmsnorm arg to prevent div by 0

#     max_batch_size: int = 32
#     max_seq_len: int = 2048
class ModelArgs:
    # Defines the model args class which contains the model's parameters, tokenizer stuff, and training parameters
    tokenizer: str = "llama"
    tokenizer_kwargs: dict = {"model_path": "tokenizer_data/tokenizer.model"}

    # tokenizer: str = "tiktoken"
    # tokenizer_kwargs: dict = {"encoding_name": "cl100k_base"}

    dim: int = 512  # size of embeddings going between transformer blocks
    n_layers: int = 8   # number of transformer blocks(attention + feed forward)
    n_heads: int = 8   # number of attention heads. Each head will have dimension head_dim = dim/n_heads
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256   # don't quite understand exactly what this is, but the hidden dimension of the ff layer will
                             # be will be a power of 2 close to 2/3 * (4dim) * (dim/multiple_of)
    norm_eps: float = 1e-5  # rmsnorm arg to prevent div by 0

    max_batch_size: int = 32
    max_seq_len: int = 512
    max_chunk_size: int = 512  # number of documents to load at a time in memory

    dir_train: str = "train_data/"
    dir_val: str = "val_data/"


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads    # attention head dim

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
        )

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], start_pos: Optional[int] = None):

        # batch x seq_len x dim
        bsz, seqlen, _ = x.shape
        # linear transformations to get Q, W, V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # split each Q, W, V into Q_i, W_i, V_i for each attention head.
        # xq, xk, xv are tensors of batch x seq_len x n_heads(n local heads probably has to do with parallel) x head_dim
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        # apply rotary(see paper for explanation) positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # only use caching during inference
        if start_pos is not None:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            keys = xk
            values = xv

        xq = xq.transpose(1, 2)  # batch x n_heads x seq_len x head_dim
        keys = keys.transpose(1, 2)   # batch x n_heads x seq_len x head_dim
        values = values.transpose(1, 2)  # batch x n_heads x seq_len x head_dim

        # QK^T/sqrt(head_dim) results in batch x n_heads x seq_len x seq_len
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # compute scores
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)  # (bs, slen, head_dim * n_heads)

        # final linear layer
        return self.wo(output)


class FeedForward(nn.Module):

    def __init__(
        self,
        dim: int,   # token embedding
        hidden_dim: int,   # this is set to 4 * dim?
        multiple_of: int,   # this is some magic
    ):
        super().__init__()

        # but hidden dim will be a power of 2 close to 2/3 * (4dim) * (dim/multiple_of)
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False,
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True,
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False,
        )

    # so SwiGLU is the operation inside self.w2. So SwiGLU(x) = Swish_activation(W_1x) * W3x
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):

        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    # transformer operation is (x -> rms_norm -> (attention + x) = h -> rms_norm -> ff + h
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], start_pos: Optional[int] = None):
        h = x + self.attention.forward(self.attention_norm(x), start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim,
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False,
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    # @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, labels: Optional[torch.Tensor] = None, start_pos: Optional[int] = None):
        _bsz, seqlen = tokens.shape

        # if labels exists, train
        # TODO: factor code here
        if labels is not None:
            output = torch.zeros((seqlen, self.vocab_size))
            # shift labels
            labels = labels[:, 1:].contiguous()
            running_loss = 0.0

            # go through each token in a sequence and train; predict the next word
            for start_pos in range(seqlen-1):
                h = self.tok_embeddings(tokens)
                self.freqs_cis = self.freqs_cis.to(h.device)
                freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

                mask = None
                if seqlen > 1:
                    mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
                    mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

                for layer in self.layers:
                    h = layer(h, freqs_cis, mask)
                h = self.norm(h)

                # output is batch size x vocab size (predictions)
                output = self.output(h[:, -1, :]).contiguous() # only compute last logits

                # calculate loss of predicting the next word:
                #   input: batch size x vocab size predictions
                #   target: batch size list of indices of correct class
                loss_function = nn.CrossEntropyLoss()
                loss = loss_function(output, labels[:,start_pos]) 

                running_loss += loss

            # return mean cross entropy loss for next token predictions for this batch of sequences
            mean_loss = running_loss / (seqlen - 1)
            return mean_loss, output

        # keeping all inference code from before
        else:
            _bsz, seqlen = tokens.shape
            h = self.tok_embeddings(tokens)
            self.freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

            mask = None
            if seqlen > 1:
                mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
                mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

            for layer in self.layers:
                h = layer(h, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)
            h = self.norm(h)
            output.append(self.output(h[:, -1, :]))  # only compute last logits
            return output
