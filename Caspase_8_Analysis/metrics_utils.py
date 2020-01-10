from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Helper function that prints the confusion matrix
def plot_confusion(classifier, title, test_X, test_Y, class_labels, show_plot, print_confusion):
    disp = plot_confusion_matrix(classifier, test_X, test_Y,
                                 display_labels=class_labels,
                                 cmap=plt.cm.Blues, normalize=None)
    disp.ax_.set_title(title)
    if (show_plot):
        plt.show()
    if (print_confusion):
        print(title)
        print(disp.confusion_matrix)


# Print the difference of probabilities
def plot_probabilities(probsList, test_Y, class_labels, show_plot, print_probabilities):
    #probs = np.array(classifier.predict_proba(test_X)).transpose()
    probs = np.array(probsList).transpose()
    y_labels = ["Active" if i != 0 else "Inactive" for i in test_Y]

    if(print_probabilities):
        print(probs.transpose())

    if(show_plot):
        plt.plot()
        plt.plot(np.arange(len(probs[0])), probs[0], label=class_labels[0], marker='o')
        plt.plot(np.arange(len(probs[0])), probs[1], label=class_labels[1], marker='o')
        plt.xticks(np.arange(len(probs[0])), y_labels, rotation='60')
        plt.xlabel('Label')
        plt.ylabel('Probability')
        plt.title('Difference of Probability (each datapoint)')
        plt.legend()
        plt.tight_layout()
        plt.show()

def doEvaluations(true_y, pred_y, pred_probs=None, probs_exist=False, displayed_name="Metrics", show_plots=False):
    print(displayed_name + " | Accuracy: " + str(metrics.accuracy_score(true_y, pred_y)))

    print(displayed_name + " | Recall: " + str(metrics.recall_score(true_y, pred_y)))

    print(displayed_name + " | F1: " + str(metrics.f1_score(true_y, pred_y, average='binary')))

    fpr, tpr, thresholds = metrics.roc_curve(true_y, pred_y)
    print(displayed_name + " | AUC: " + str(metrics.auc(fpr, tpr)))

    if(probs_exist):
        plot_probabilities(pred_probs, true_y, ["Inactive", "Active"], show_plot=show_plots,
                            print_probabilities=False)