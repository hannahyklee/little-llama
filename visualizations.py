
import numpy as np
import pandas as pd
import seaborn as sns
import json
import sys
from matplotlib import pyplot as plt


def plot_loss(filepath: str):
    f = open(filepath)
    data = json.load(f)

    log_history = data['log_history']
    # log_history is a list of dictionaries, {train},{eval},{train},{eval},...
    results = {'train_losses':[], 
               'val_losses':[],
               'Steps':[]}

    for log in log_history:
        if 'loss' in log.keys():
            results['train_losses'].append(log['loss'])
            results['Steps'].append(log['step'])
        if 'eval_loss' in log.keys():
            results['val_losses'].append(log['eval_loss'])

    df = pd.DataFrame(data=results)
    dfm = df.melt(["Steps"], var_name="Dataset", value_name="Loss")

    # plot train and validation loss
    sns.lineplot(data=dfm, x='Steps', y="Loss", hue='Dataset', legend='full').set(title="")
    ax = plt.gca()
    plt.savefig('loss_plot.png')
                

def main():
    filepath = sys.argv[1]
    plot_loss(filepath)

    # ./test_model_dir/checkpoint-1000/trainer_state.json

if __name__ == "__main__":
    main()
