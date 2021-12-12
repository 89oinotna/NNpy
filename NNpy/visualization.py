import matplotlib.pyplot as plt
import numpy as np


def plot(tr_loss, vl_loss, tr_metric, vl_metric):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(np.array(tr_loss), 'b-', label='Train Loss')
    ax1.plot(np.array(vl_loss), 'r--', label='Valid Loss')
    ax1.set(xlabel='Ephocs', ylabel='Loss')
    ax1.set_title('MSE', fontsize=15)
    ax1.legend(prop={'size': 15})

    ax2.plot(np.array(tr_metric), 'b-', label='Train Accuracy')
    ax2.plot(np.array(vl_metric), 'r--', label='Valid Accuracy')
    ax2.set(xlabel='Ephocs', ylabel='Accuracy')
    ax2.set_title('Accuracy', fontsize=15)
    ax2.legend(prop={'size': 15})
    plt.gcf().set_size_inches((20, 5), forward=False)

    plt.show()
