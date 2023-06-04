import torch
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama_main.llama.model import ModelArgs, Transformer
from train import setup_model_args
from our_tokenizers import init_tokenizer
from llama_main.llama.generation import LLaMA


# what is our test set? I think just some random phrases we come up with and tokenize

# use the llama tokenizer

def make_prompts():
    """
    Returns list of prompts
    """
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
    return prompts


if __name__ == "__main__":
    # set up system to be able to run the model
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")  # "nccl" for gpu, "gloo" if on windows/mac
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # set up the model
    model_args = setup_model_args()
    model = Transformer(model_args) # we will see if it works to setup the model like this - it might not have all the info it needs

    checkpoint = torch.load("test_model_dir/checkpoint-12000/pytorch_model.bin")
    model.load_state_dict(checkpoint["model_state_dict"])

    tokenizer = init_tokenizer(model_args)
    llama = LLaMA(model=model, tokenizer=tokenizer)   

    # I think I can just call generate from LLaMA at this point
    # TODO finish 

    # # tokenize text
    # prompts = ["What is the meaning of life?"]
    # llama_tok = init_tokenizer(model_args)
    # tokens = llama_tok.encode_batch(prompts, train=False)
    # # note we may have to convert tokens to pytorch tensors


    # model = AutoModelForSequenceClassification.from_pretrained("output_dir")

    # attempting to load the pytorch model bin file. But that makes the output difficult to handle 
    # in the last checkpoint for each a run, there appears to be a pytorch_model.bin file
    # we load the model state from this file
    # checkpoint = torch.load("test_model_dir/checkpoint-12000/pytorch_model.bin")
    # model.load_state_dict(checkpoint["model_state_dict"])

    # make inference
    # model.eval()
    # output = model(**tokens) # this out output a distribution


