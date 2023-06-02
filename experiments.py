from train import train
import sys

# set training arguments
training_args_baseline = {
    'output_dir': f'experiment_data/test_model_dir',
    'lr_scheduler_type': 'cosine',
    'warmup_steps': 2000,
}

training_args_cosine_no_warmup = {
    'output_dir': f'experiment_data/cosine_no_warmup',
    'lr_scheduler_type': 'cosine',
    'warmup_steps': 0,
}

training_args_linear_warmup = {
    'output_dir': f'experiment_data/linear_warmup',
    'lr_scheduler_type': 'linear',
    'warmup_steps': 2000,
}

def main():
    training_args = None
    # get the correct experiment 
    experiment = sys.argv[1] 
    if experiment == 'baseline':
        training_args = training_args_baseline
    elif experiment == 'cosine_nowarmup':
        training_args = training_args_cosine_no_warmup
    elif experiment == 'linear_warmup':
        training_args = training_args_linear_warmup

    # run training
    train(training_args)

if __name__ == "__main__":
    main()