from train import train
import sys


def main():
    """
    Sets up an experiment and runs training for the LLaMA model.

    Takes in 3 arguments: the type of learning rate scheduler, the number of warmup steps, and the number of epochs
    to train for.

    Run with:
        ```
        torchrun experiments.py <cosine/linear> <num warmup steps> <epochs>
        ```
    """
    training_args = {}
    # get the correct experiment setup
    if (len(sys.argv) != 4):
        print(f"Expected 3 arguments to experiments.py; received {len(sys.argv)}")
        exit(1)

    scheduler = sys.argv[1] 
    warmup_steps = sys.argv[2]
    epochs = sys.argv[3]

    scheduler_options = ['cosine', 'linear']

    if scheduler not in scheduler_options:
        print(f'Unexpected scheduler found; expected one of {scheduler_options}')
        exit(1)
    
    # set parameters
    training_args['lr_scheduler_type'] = scheduler
    training_args['warmup_steps'] = int(warmup_steps)
    training_args['epochs'] = int(epochs)
    training_args['output_dir'] = f'experiment_data/{scheduler}_{warmup_steps}_{epochs}'

    # run training
    train(training_args)

if __name__ == "__main__":
    main()