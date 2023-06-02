from train import train
import sys

def main():
    training_args = {}
    # get the correct experiment 
    scheduler = sys.argv[1] 
    warmup_steps = sys.argv[2]
    epochs = sys.argv[3]

    scheduler_options = ['cosine', 'linear']

    if scheduler not in scheduler_options:
        print(f'Unexpected scheduler found; expected one of {scheduler_options}')
    
    training_args['lr_scheduler_type'] = scheduler
    training_args['warmup_steps'] = int(warmup_steps)
    training_args['epochs'] = int(epochs)
    training_args['output_dir'] = f'experiment_data/{scheduler}_{warmup_steps}_{epochs}'

    # run training
    train(training_args)

if __name__ == "__main__":
    main()