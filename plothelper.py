import matplotlib.pyplot as plt
import numpy as np


class plothelper():
    @staticmethod
    def plot(x_axis, train_scores, val_scores, label_training, label_validation):
        plt.close()
        plt.figure()
        train_scores_mean = np.mean(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        train_plot = plt.plot(x_axis, train_scores_mean, '-o', markersize=2, label=label_training)
        val_plot = plt.plot(x_axis, val_scores_mean, '-o', markersize=2, label=label_validation)
