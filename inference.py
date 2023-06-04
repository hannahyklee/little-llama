import torch
import os
import sys

from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama_main.llama.model import ModelArgs, Transformer
from train import setup_model_args
from our_tokenizers import init_tokenizer
from llama_main.llama.generation import LLaMA
from loader import DataLoader # importing just so that we can set up Transformer


"""
This script perfoms inference on a saved LLaMA model
Arguments:
[1] path of the directory where saved model pytorch_model.bin can be found
"""

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
    # Parameter: [1] path of the directory where saved model pytorch_model.bin can be found
    dir = sys.argv[1]
    model_path = os.path.join(dir, 'pytorch_model.bin')

    # set up system to be able to run the model
    assert torch.cuda.is_available()
    DEVICE = "cuda"
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    print('set world size etc')

    torch.distributed.init_process_group("nccl")  # "nccl" for gpu, "gloo" if on windows/mac
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)
    print('initialized cuda')

    # set up the model
    train_args = setup_model_args()
    train_dl = DataLoader(train_args, train=True) # set up training dataloader for convenience so train_args gets set up correctly 
    model = Transformer(train_args) 
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)

    tokenizer = init_tokenizer(train_args)
    llama = LLaMA(model=model, tokenizer=tokenizer) 

    # generate 
    prompts = make_prompts()
    outputs = llama.generate(prompts=prompts, max_gen_len=128)  

    for i in range(len(prompts)):
        print("Prompt:", prompts[i])
        print("Response:", outputs[i])

   