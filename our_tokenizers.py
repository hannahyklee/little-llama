import tiktoken
from llama_main.llama.tokenizer import Tokenizer as LlamaTokenizer
from llama_main.llama.model import ModelArgs

from typing import List, Iterable


class TokenizerWrapper:
    # class that wraps actual tokenizer and implements a common interface
    def __init__(self):

        self._tokenizer = None  # the actual tokenizer class

        self.n_words: int = -1  # vocab_size
        self.bos_id: int = -1   # begin_of_sentence id
        self.eos_id: int = -1   # end_of_sentence id

    # encode one string
    # train indicates whether the batch is for training or validation(True) or for generation(False)
    # if train = True, add bos and eos to tokenized documents, otherwise add just bos
    def encode(self, s: str, train: bool) -> List[int]:
        raise NotImplementedError

    # encodes a whole batch of strings
    def encode_batch(self, b_of_s: List[str], train: bool) -> List[List[int]]:
        return [self.encode(s, train) for s in b_of_s]

    # used only for generation.
    # decodes a list of tokens. Assumes the list of tokens is of the form [bos_id, tokens, eos_id, potentially padding]
    def decode(self, t: List[int]) -> str:

        assert type(t) is list

        # remove bos
        t = t[1:]
        # reduce list to up to eos tok if any
        try:
            t = t[:t.index(self.eos_id)]
        except ValueError:
            pass

        return self._tokenizer.decode(t)

    # decodes a batch of lists of tokens
    def decode_batch(self, b_of_t: List[List[int]]) -> List[str]:
        return [self.decode(t) for t in b_of_t]


class TikTokenWrapper(TokenizerWrapper):

    def __init__(self, encoding_name: str = "cl100k_base"):

        super(TikTokenWrapper).__init__()

        self._tokenizer = tiktoken.get_encoding(encoding_name)

        self.n_words: int = self._tokenizer.n_vocab
        self.bos_id: int = self.n_words + 1
        self.eos_id: int = self.n_words + 2

    def encode(self, s: str, train: bool) -> List[int]:

        assert type(s) is str
        if train:
            return [self.bos_id] + self._tokenizer.encode(s) + [self.eos_id]
        else:
            return [self.bos_id] + self._tokenizer.encode(s)


class LlamaWrapper(TokenizerWrapper):

    def __init__(self, model_path: str):

        super(LlamaWrapper).__init__()

        self._tokenizer = LlamaTokenizer(model_path=model_path)

        self.n_words: int = self._tokenizer.n_words
        self.bos_id: int = self._tokenizer.bos_id
        self.eos_id: int = self._tokenizer.eos_id

    def encode(self, s: str, train: bool) -> List[int]:
        return self._tokenizer.encode(s, bos=True, eos=train)


# use this function to create tokenizers
def init_tokenizer(params: ModelArgs) -> TokenizerWrapper:

    if params.tokenizer == "tiktoken":
        return TikTokenWrapper(**params.tokenizer_kwargs)
    elif params.tokenizer == "llama":
        return LlamaWrapper(**params.tokenizer_kwargs)
    else:
        raise ValueError("Tokenizer can be tiktoken or llama.")


# some test code
if __name__ == "__main__":

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        """Tweet: "I hate it when my phone battery dies."
        Sentiment: Negative
        ###
        Tweet: "My day has been 👍"
        Sentiment: Positive
        ###
        Tweet: "This is the link to the article"
        Sentiment: Neutral
        ###
        Tweet: "This new music video was incredibile"
        Sentiment:""",
                """Translate English to French:
        
        sea otter => loutre de mer
        
        peppermint => menthe poivrée
        
        plush girafe => girafe peluche
        
        cheese =>""",
    ]

    def test_tokenizer(tok: TokenizerWrapper, inp_batch):

        print(type(tok._tokenizer), tok.n_words)

        e_batch = tok.encode_batch(inp_batch, train=True)
        d_batch = tok.decode_batch(e_batch)

        for s, t, e in zip(inp_batch, e_batch, d_batch):
            assert t[0] == tok.bos_id
            assert t[-1] == tok.eos_id
            assert s == e

        e_batch = tok.encode_batch(inp_batch, train=False)
        d_batch = tok.decode_batch(e_batch)

        for s, t, e in zip(inp_batch, e_batch, d_batch):
            assert t[0] == tok.bos_id
            assert s == e


    model_args = ModelArgs()

    llama_tok = init_tokenizer(model_args)
    test_tokenizer(llama_tok, prompts)

    model_args.tokenizer = "tiktoken"
    model_args.tokenizer_kwargs = {"encoding_name": "cl100k_base"}

    tiktoken_tok = init_tokenizer(model_args)
    test_tokenizer(tiktoken_tok, prompts)

