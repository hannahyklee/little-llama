import numpy as np
import pandas as pd
import seaborn as sns
import json
from matplotlib import pyplot as plt


def plot_losses(filepaths: list):
    """
    Plots training and validation losses vs. steps for each model. Organizes style by train or validation sets, and 
    colors by model.
    """
    # create dataframe
    df = pd.DataFrame(columns=['Loss', 'Dataset', 'Steps', 'Model'])
    for filepath in filepaths:
        results = read_json(filepath)
        newdf = pd.DataFrame(columns=df.columns)
        for i, loss in enumerate(results['train_losses']):
            newdf = pd.concat([newdf, pd.DataFrame([{'Loss': loss, 'Dataset': 'Train', 'Steps': results['steps'][i]}])], ignore_index=True)
            newdf = pd.concat([newdf, pd.DataFrame([{'Loss': results['val_losses'][i], 'Dataset': 'Val', 'Steps': results['steps'][i]}])], ignore_index=True)
        newdf['Model'] = results['model_name']
        df = pd.concat([df, newdf])
            
    sns.lineplot(data=df, x='Steps', y='Loss', hue='Model', style='Dataset', linewidth=0.75)
    plt.show()


def plot_learning_rates(filepaths: list):
    """
    Plots learning rates vs. steps for different models.
    """
    # create dataframe
    df = pd.DataFrame(columns=['Learning Rate', 'Steps', 'Model'])
    for filepath in filepaths:
        results = read_json(filepath)
        newdf = pd.DataFrame(columns=df.columns)
        for i, lr in enumerate(results['learning_rate']):
            newdf = pd.concat([newdf, pd.DataFrame([{'Learning Rate': lr, 'Steps': results['steps'][i]}])], ignore_index=True)
        newdf['Model'] = results['model_name']
        df = pd.concat([df, newdf])
            
    sns.lineplot(data=df, x='Steps', y='Learning Rate', hue='Model')
    plt.show()


def plot_val_loss(filepaths: list):
    """
    Plots validation loss vs. compute (in floating point operations)
    """
    df = pd.DataFrame(columns=["Val Loss", "Model", "Scheduler", "Warmup Steps", "Compute"])
    for filepath in filepaths:
        results = read_json(filepath)
        min_val_loss = np.min(results["val_losses"])
        row = {"Val Loss": min_val_loss, 
                "Model": results["model_name"], 
                "Scheduler": results["scheduler"], 
                "Warmup Steps": results["warmup_steps"],
                "Compute": results["compute"]}
        df = pd.concat([df, pd.DataFrame(data=[row])], ignore_index=True)

    sns.lineplot(data=df, x="Compute", y="Val Loss", hue='Warmup Steps', style="Scheduler")
    plt.show()


def read_json(filepath: str):
    f = open(filepath)
    data = json.load(f)

    log_history = data['log_history']
    # Note: dependent on filepaths given in main(). Should be of form "scheduler_warmup_epochs".
    experiment = filepath.split('/')[-2] 

    # get scheduler type and warmup steps from experiment name
    vals = experiment.split('_')
    scheduler = vals[0]
    warmup_steps = vals[1]
    ops = data["total_flos"]

    results = {'train_losses':[], 
            'val_losses':[],
            'steps':[],
            'learning_rate': [],
            'model_name': experiment,
            'scheduler': scheduler,
            'warmup_steps': warmup_steps,
            'compute': ops}


    for log in log_history:
        if 'loss' in log.keys():
            results['train_losses'].append(log['loss'])
            results['steps'].append(log['step'])
            results['learning_rate'].append(log['learning_rate'])
        if 'eval_loss' in log.keys():
            results['val_losses'].append(log['eval_loss'])

    return results
                

def main():
    
    fpaths = [
            #   './experiment_data/cosine_0_1/trainer_state.json',
                './experiment_data/cosine_0_3/trainer_state.json',
              './experiment_data/cosine_2000_1/trainer_state.json',
              './experiment_data/cosine_2000_3/trainer_state.json',
              './experiment_data/cosine_2000_6/trainer_state.json',
              './experiment_data/linear_2000_1/trainer_state.json',
              './experiment_data/linear_2000_3/trainer_state.json',
              './experiment_data/linear_2000_6/trainer_state.json']
    
    # plot_losses(fpaths)
    # plot_learning_rates(fpaths)
    plot_val_loss(fpaths)

if __name__ == "__main__":
    main()
