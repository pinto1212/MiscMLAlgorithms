# AdaBoost class

import matplotlib.pyplot as plt

from base import base

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IMAGE_DIR = 'output/'


class AdaBoost(base):

    def __init__(self, n_estimators=50, learning_rate=1., max_depth=3, random_state=42):

        print('AdaBoosted Decision Tree classifier')
        super().__init__(modelName='AdaBoost')

        self.model = Pipeline([('scaler', StandardScaler()),
                               ('adaboost', AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth,
                                                                                      random_state=random_state),
                                                               n_estimators=n_estimators, learning_rate=learning_rate,
                                                               random_state=random_state))])

        self.default_params = {'adaboost__n_estimators': n_estimators,
                               'adaboost__learning_rate': learning_rate,
                               'adaboost__base_estimator__max_depth': max_depth}

    def ValidationCurve(self, x_train, y_train, **kwargs):

        print('\n\nModel Complexity Analysis')
        self.optimal_params = self.default_params.copy()

        plt.figure()
        kwargs['param'] = 'adaboost__base_estimator__max_depth'
        kwargs['range'] = kwargs['max_depth_range']
        kwargs['xlabel'] = 'Maximum depth'
        kwargs['xscale'] = 'linear'
        kwargs['label_training'] = 'Training'
        kwargs['label_validation'] = 'Validation'

        best_max_depth, score = super().ValidationCurve(x_train, y_train, **kwargs)
        print('--> max_depth = {} --> score = {:.4f}'.format(best_max_depth, score))

        self.optimal_params['adaboost__base_estimator__max_depth'] = best_max_depth
        plt.savefig(IMAGE_DIR + '{}_max_depth'.format(self.modelName))

        plt.figure()
        kwargs['param'] = 'adaboost__n_estimators'
        kwargs['range'] = kwargs['n_estimators_range']
        kwargs['xlabel'] = 'estimators'

        best_n_estimators, score = super().ValidationCurve(x_train, y_train, **kwargs)
        print('--> n_estimators = {} --> score = {:.4f}'.format(best_n_estimators, score))

        self.optimal_params['adaboost__n_estimators'] = best_n_estimators
        plt.savefig(IMAGE_DIR + '{}_n_estimators'.format(self.modelName))

        plt.figure()
        kwargs['param'] = 'adaboost__learning_rate'
        kwargs['range'] = kwargs['learning_rate_range']
        kwargs['xlabel'] = 'Learning Rate'
        kwargs['xscale'] = 'log'

        best_learning_rate, score = super().ValidationCurve(x_train, y_train, **kwargs)
        print('--> learning_rate = {} --> score = {:.4f}'.format(best_learning_rate, score))

        self.optimal_params['adaboost__learning_rate'] = best_learning_rate
        plt.savefig(IMAGE_DIR + '{}_learning_rate'.format(self.modelName))

        print('\nBest params have been found to be :\n {}'.format(self.optimal_params))
        self.model.set_params(**self.optimal_params)
