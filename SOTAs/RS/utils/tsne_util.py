import matplotlib.pyplot as plt
import numpy as np

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    unilabel = np.unique(label)
    num_class = len(unilabel)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    
    # setting colors for seperating the label and unlabel samples 
    colors = ['r','chocolate','orange','gold','y','g','c','deepskyblue','b','m'] 
    
    for i in range(num_class):
        idx = np.where(label == unilabel[i])
        plt.scatter(data[idx, 0], data[idx, 1], color=colors[i], s = 5)
    plt.legend(['class = {}'.format(i) for i in unilabel])
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig