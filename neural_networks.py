import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from base import base
from plothelper import plothelper

IMAGE_DIR = 'output/'


class NeuralNetwork(base):

    def __init__(self, alpha=2.395, layer1_nodes=50, layer2_nodes=30, learning_rate=0.001, max_iter=200, activation='relu'):
        super().__init__(modelName='ANN')
        self.model = Pipeline([('scaler', StandardScaler()),
                               ('nn', MLPClassifier(hidden_layer_sizes=(layer1_nodes, layer2_nodes), activation=activation,
                                                    solver='sgd', alpha=alpha, batch_size=200, learning_rate='constant',
                                                    learning_rate_init=learning_rate, max_iter=max_iter, tol=1e-8,
                                                    early_stopping=False, validation_fraction=0.1, momentum=0.5,
                                                    n_iter_no_change=max_iter, random_state=42))])

        self.default_params = {'nn__alpha': alpha,
                               'nn__learning_rate_init': learning_rate}
        self.activation = activation

    def plot_confusion_matrix(self,cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.coolwarm):
        return super().plot_confusion_matrix(cm,classes,normalize,title+' - {}'.format(self.activation), cmap)
            

    def ValidationCurve(self, x_train, y_train, **params):
        self.optimal_params = self.default_params.copy()
        plt.figure()
        params['param'] = 'nn__learning_rate_init'
        params['range'] = params['learning_rate_range']
        params['xlabel'] = 'Learning Rate'
        params['xscale'] = 'log'
        params['label_training'] = 'Training'
        params['label_validation'] = 'Validation'
        params['title'] = 'Validation curve - ANN - {}'.format(self.activation)

        best_learning_rate, score = super().ValidationCurve(x_train, y_train, **params)
        print('--> best_learning_rate = {} --> score = {:.4f}'.format(best_learning_rate, score))

        self.optimal_params['nn__learning_rate_init'] = best_learning_rate
        plt.savefig(IMAGE_DIR + '{}_learning_rate'.format(self.modelName))

        params['param'] = 'nn__alpha'
        params['range'] = params['alpha_range']
        params['xlabel'] = r'L2 regularization term $\alpha$'

        best_alpha, score = super().ValidationCurve(x_train, y_train, **params)
        print('--> best_alpha = {} --> score = {:.4f}'.format(best_alpha, score))

        self.optimal_params['nn__alpha'] = best_alpha

        print('\nBest params have been found to be :\n {}'.format(self.optimal_params))
        self.model.set_params(**self.optimal_params)

    def LearningCurve(self, x_train, y_train, **params):
        super().LearningCurve(x_train, y_train, **params)

        max_iter = self.model.get_params()['nn__max_iter']
        epochs = np.arange(1, max_iter + 1, 1)
        train_scores, val_scores = [], []


        model = clone(self.model)
        model.set_params(**{'nn__max_iter': 1, 'nn__warm_start': True})


        # Perform k-fold Stratified cross-validation
        for train_fold, val_fold in StratifiedKFold(n_splits=params['cv']).split(x_train, y_train):

            # Training and validation set for current fold
            x_train_fold, x_val_fold = x_train[train_fold], x_train[val_fold]
            y_train_fold, y_val_split = y_train[train_fold], y_train[val_fold]

            # List of training and validation scores for current fold
            train_scores_fold, val_scores_fold = [], []

            # Loop through epochs
            for _ in epochs:
                # Fit model on the training set of the current fold
                model.fit(x_train_fold, y_train_fold)

                # Append training and validation score to corresponding lists
                train_scores_fold.append(1-model.score(x_train_fold, y_train_fold))
                val_scores_fold.append(1-model.score(x_val_fold, y_val_split))

            # Append training and validation scores of current fold to corresponding lists
            train_scores.append(train_scores_fold)
            val_scores.append(val_scores_fold)

        # Convert to numpy arrays to plot
        train_scores = np.array(train_scores)
        val_scores = np.array(val_scores)

        # Plot training and validation scores vs. epochs
        plt.close()
        plt.figure()
        plothelper.plot(epochs, train_scores.T, val_scores.T, label_training='Training', label_validation='Validation')

        # Add title, legend, axes labels and eventually set y axis limits
        plt.title('ANN- Error per iteration'.format(params['cv']))
        plt.legend(loc='best')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.ylim(-0.01, 0.2)

        # Save figure
        plt.savefig(IMAGE_DIR + '{}_epochs'.format(self.modelName))
