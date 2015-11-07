import matplotlib.pyplot as plt

from base import base

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IMAGE_DIR = 'output/'


class KNN(base):

    def __init__(self, k=3, weights='distance', p=2):

        super().__init__(modelName='kNN')

        self.defaultparams = {'knn__n_neighbors': k,
                               'knn__weights': weights,
                               'knn__metric': 'minkowski',
                               'knn__p': p}

        self.model = Pipeline([('scaler', StandardScaler()),
                               ('knn', KNeighborsClassifier(n_neighbors=k,
                                                            weights=weights,
                                                            metric='minkowski',
                                                            p=p,
                                                            n_jobs=-1))])

    def ValidationCurve(self, xtrain, ytrain, **params):
        self.bestparams = self.defaultparams.copy()

        besterror, bestp, bestfct, bestk = 0., 1, '', 1

        plt.figure()
        self.SetParamsForK(params)

        # browse all functions
        for fct in params['weight_functions']:

            self.bestparams['knn__weights'] = fct

            params['label_training'] = 'Training'
            params['label_validation'] = 'Validation'

            k, score = super().ValidationCurve(xtrain, ytrain, **params)
            print('--> k = {}, weight = {} --> score = {:.4f}'.format(k, fct, score))

            if score > besterror:
                besterror, bestk, bestfct = score, k, fct

        self.bestparams['knn__n_neighbors'] = bestk
        self.bestparams['knn__weights'] = bestfct
        plt.savefig(IMAGE_DIR + '{}_k'.format(self.modelName))

        plt.figure()
        self.SetParamForP(params)

        bestp, score = super().ValidationCurve(xtrain, ytrain, **params)

        self.bestparams['knn__p'] = bestp
        plt.savefig(IMAGE_DIR + '{}_p'.format(self.modelName))
        self.model.set_params(**self.bestparams)

    def SetParamForP(self, params):
        params['param'] = 'knn__p'
        params['range'] = params['p_range']
        params['xlabel'] = 'Power parameter p'
        params['label_training'] = 'Training'
        params['label_validation'] = 'Validation'

    def SetParamsForK(self, params):
        params['param'] = 'knn__n_neighbors'
        params['range'] = params['n_neighbors_range']
        params['xlabel'] = 'Number of neighbors k'
        params['xscale'] = 'linear'
