#!/usr/bin/env python3.7
import subprocess
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

def moving_average(data, window):
    data = np.cumsum(data, dtype=float)
    data[window:] = data[window:] - data[:-window]
    return data[window - 1:] / window

def collect_data():
    compile_net = subprocess.run(['./build-release.sh'], stdout=subprocess.PIPE, shell=True, check=True)
    run_net = subprocess.run(['./run-release.sh | tail -n +5'], stdout=subprocess.PIPE, shell=True, check=True)
    output = str(run_net.stdout).split('\\n')[:-1]
    data = [float(line.split(' ')[1]) for line in output]
    data = np.array(data)
    data = moving_average(data, 1000)
    seaborn_data = {
        'training_example': [],
        'cost': [],
    }
    for index, data in enumerate(data):
        seaborn_data['training_example'].append(index)
        seaborn_data['cost'].append(data)
    return seaborn_data

def plot_data(costs):
    plot = sns.lineplot(x='training_example', y='cost', data=costs)
    plot.set(ylabel='Cost', xlabel='Training example')
    plt.show()

def main():
    plot_data(collect_data())

if __name__ == '__main__':
    main()
