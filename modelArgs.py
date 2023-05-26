from paths import dir_token, dir_train, dir_val


class ModelArgs:
    # Defines the model args class which contains the model's parameters, tokenizer stuff, and training parameters
    tokenizer: str = "llama"
    tokenizer_kwargs: dict = {"model_path": dir_token + "tokenizer.model"}

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

    dir_train: str = dir_train
    dir_val: str = dir_val



