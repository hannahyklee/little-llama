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
    sns.set_theme()
    df = pd.DataFrame(columns=['Loss', 'Dataset', 'Steps', 'Model', 'Schedule'])
    for filepath in filepaths:
        results = read_json(filepath)
        newdf = pd.DataFrame(columns=df.columns)
        for i, loss in enumerate(results['train_losses']):
            newdf = pd.concat([newdf, pd.DataFrame([{'Loss': loss, 
                                                     'Dataset': 'Train', 
                                                     'Steps': results['steps'][i], 
                                                     'Schedule': results['scheduler']}])], ignore_index=True)
            newdf = pd.concat([newdf, pd.DataFrame([{'Loss': results['val_losses'][i], 
                                                     'Dataset': 'Val', 
                                                     'Steps': results['steps'][i], 
                                                     'Schedule': results['scheduler']}])], ignore_index=True)
        newdf['Model'] = results['model_name']
        df = pd.concat([df, newdf])
            
    sns.lineplot(data=df, x='Steps', y='Loss', hue='Schedule', style='Dataset')
    plt.title("Loss vs. Steps")
    plt.show()


def plot_learning_rates(filepaths: list):
    """
    Plots learning rates vs. steps for different models.
    """
    # create dataframe
    df = pd.DataFrame(columns=['Learning Rate', 'Steps', 'Model', 'Schedule'])
    for filepath in filepaths:
        results = read_json(filepath)
        newdf = pd.DataFrame(columns=df.columns)
        for i, lr in enumerate(results['learning_rate']):
            newdf = pd.concat([newdf, pd.DataFrame([{'Learning Rate': lr, 'Steps': results['steps'][i], 'Schedule': results['scheduler']}])], ignore_index=True)
        newdf['Model'] = results['model_name']
        df = pd.concat([df, newdf])
            
    sns.lineplot(data=df, x='Steps', y='Learning Rate', hue='Schedule', style='Model')
    plt.title("Learning Rate vs. Steps")
    plt.show()


def plot_val_loss(filepaths: list):
    """
    Plots validation loss vs. compute (in floating point operations)
    """
    df = pd.DataFrame(columns=["Val Loss", "Model", "Schedule", "Warmup Steps", "Compute"])
    min_losses = {}
    for filepath in filepaths:
        results = read_json(filepath)
        min_val_loss = np.min(results["val_losses"])
        row = {"Val Loss": min_val_loss, 
                "Model": results["model_name"], 
                "Schedule": results["scheduler"], 
                "Warmup Steps": results["warmup_steps"],
                "Compute": results["compute"]}
        df = pd.concat([df, pd.DataFrame(data=[row])], ignore_index=True)
        # keep track of min losses for each compute level
        if results["compute"] not in min_losses.keys():
            min_losses[results["compute"]] = [min_val_loss]
        else:
            min_losses[results["compute"]].append(min_val_loss)

    sns.lineplot(data=df, x="Compute", y="Val Loss", style="Schedule")
    plt.xlabel("Compute (Floating Point Operations)")
    plt.title("Scaling: Validation Loss vs. Compute")
    plt.show()

    # plot differences 
    diff_df = pd.DataFrame(columns=['Compute', "Difference"])
    for compute in min_losses.keys():
        row = {"Compute": compute, "Difference": np.abs(min_losses[compute][0] - min_losses[compute][1])}
        diff_df = pd.concat([diff_df, pd.DataFrame(data=[row])], ignore_index=True)

    sns.lineplot(data=diff_df, x="Compute", y="Difference")
    plt.xlabel("Compute (Floating Point Operations)")
    plt.title("Difference in Validation Loss")
    plt.show()

def read_json(filepath: str):
    """
    Reads contents from a given file and populates a dictionary with result statistics, including:
    training losses, validation losses, steps at which the losses were calculated, learning rates,
    and the experiment settings currently being used.
    """
    f = open(filepath)
    data = json.load(f)

    log_history = data['log_history']
    # Note: dependent on filepaths given in main(). Should be of form "scheduler_warmup_epochs".
    experiment = filepath.split('/')[-2] 

    # get scheduler type and warmup steps from experiment name
    vals = experiment.split('_')
    scheduler = vals[0]
    scheduler = scheduler.capitalize()
    warmup_steps = vals[1]
    ops = data["total_flos"]

    # rename experiment
    experiment = scheduler + ', ' + vals[2] + ' Epochs'

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
              './experiment_data/cosine_2000_10/trainer_state.json',
              './experiment_data/linear_2000_1/trainer_state.json',
              './experiment_data/linear_2000_3/trainer_state.json',
              './experiment_data/linear_2000_6/trainer_state.json',
              './experiment_data/linear_2000_10/trainer_state.json']
    
    plot_losses(fpaths)
    plot_learning_rates(fpaths)
    plot_val_loss(fpaths)

if __name__ == "__main__":
    main()
