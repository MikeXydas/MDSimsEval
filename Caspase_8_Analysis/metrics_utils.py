from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

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


# Print the difference of probabilities from predict_proba of each class
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


# Run metrics for Accuracy, Recall, F1, AUC and print difference in probability predictions (if possible)
def do_evaluations(true_y, pred_y, pred_probs=None, probs_exist=False, displayed_name="Metrics", show_plots=False):
    print(displayed_name + " | Accuracy: " + str(metrics.accuracy_score(true_y, pred_y)))

    print(displayed_name + " | Recall: " + str(metrics.recall_score(true_y, pred_y)))

    print(displayed_name + " | F1: " + str(metrics.f1_score(true_y, pred_y, average='binary')))

    # print(pred_probs)
    if(probs_exist):
        fpr, tpr, thresholds = metrics.roc_curve(true_y, pred_probs[:, 1], pos_label=1)
        print(displayed_name + " | AUC: " + str(metrics.auc(fpr, tpr)))
        plot_probabilities(pred_probs, true_y, ["Inactive", "Active"], show_plot=show_plots,
                            print_probabilities=False)


# Plotting a 2D graph of scattered points
# plotted_df is X:Y, the column names will be used as axis
# X must have 2 columns, Y must be one column (so potted_df 3 columns)
# If -1 returned the plot failed (due to wrong dimensions given)
# TODO: Functionality for 1D and 3D plots
def plot_scatter_points(plotted_df, targets, unique_labels=["label1", "axis_2"], title="Plot", show_ids=True):
    if(len(plotted_df.columns) != 3):
        return -1

    plt.plot()
    plt.xlabel(plotted_df.columns[0], fontsize=20)
    plt.ylabel(plotted_df.columns[1], fontsize=20)
    plt.title(title, fontsize=25)
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    colors = colors[:len(targets)]

    X_df = plotted_df.iloc[:, :-1]
    Y_df = plotted_df.iloc[:, -1]
    #print(X_df)
    X_df = pd.concat([X_df, pd.DataFrame(np.arange(len(X_df.index)))], axis=1)

    for target, color in zip(targets, colors):
        indices_to_keep = Y_df.values == target
        plt.scatter(X_df.iloc[indices_to_keep, 0]
                    , X_df.iloc[indices_to_keep, 1]
                    , c=color
                    , s=50)

        # Adding ids on the plotted points
        if(show_ids):
            for i in range(sum(indices_to_keep)):
                plt.annotate(str(X_df.iloc[indices_to_keep, 2].iloc[i]),
                             (X_df.iloc[indices_to_keep, 0].iloc[i], X_df.iloc[indices_to_keep, 1].iloc[i]),
                             weight='bold')

    plt.legend(unique_labels)
    plt.grid()

    plt.show()

    return 0


def performKFold(classifier, X, Y, folds):
    kf = KFold(n_splits=folds, shuffle=True)
    acc = f1 = auc = rec = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        classifier.fit(X_train, y_train)
        pred_y = classifier.predict(X_test)
        probs = classifier.predict_proba(X_test)[:, 1]
        acc += metrics.accuracy_score(y_test, pred_y)
        f1 += metrics.f1_score(y_test, pred_y)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, probs, pos_label=1)
        auc += metrics.auc(fpr, tpr)
        rec += metrics.recall_score(y_test, pred_y)

    print("\n>>> " + str(folds) + "-Fold Results")
    print(str(folds) + "-Fold | Accuracy: " + str(acc / folds))
    print(str(folds) + "-Fold | Recall: " + str(rec / folds))
    print(str(folds) + "-Fold | F1: " + str(f1 / folds))
    print(str(folds) + "-Fold | AUC: " + str(auc / folds))
