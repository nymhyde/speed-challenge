import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D


def movingAverage(x, window):

    ret = np.zeros_like(x)

    for i in range(len(x)):
        id1 = max(0, i-(window-1)//2)
        id2 = min(len(x), i+(window-1)//2 + (2 - (window%2)))

        ret[i] = np.mean(x[id1:id2])

    return ret


def computeAverage(x, window, idx):

    min_idx = max(0, idx - window - 1)
    return np.mean(x[min_idx:idx])


def plot(predict_values, gt):

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(gt)), gt, label = 'ground truth')
    ax.plot(np.arange(len(predict_values)), np.array(predict_values), label = 'predict')
    start, end = ax.get_xlim()
    ax.yaxis.set_ticks(np.arange(0, max(gt)+10, 5.0))
    ax.legend(loc='upper left')
    
    plt.xlabel('frame num.')
    plt.ylabel('speed [m/s]')
    ax.figure.savefig('result.png', format='png', dpi=1200)
    plt.show()
