# Support Vector Machines class

import matplotlib.pyplot as plt

from base import base

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IMAGE_DIR = 'output/'


class SVM(base):

    def __init__(self, c=1., kernel='rbf', degree=3, gamma=0.001, random_state=42):
        """Initialize Support Vector Machines (SVM).

            Args:
                c (float): regularization term.
                kernel (string): kernel function.
                degree (int): degree for polynomial kernels.
                gamma (float): gamma value for rbf and tanh kernels.
                random_state (int): random seed.

            Returns:
                None.
            """

        # Initialize Classifier
        print('SVM classifier')
        super().__init__(modelName='SVM')

        # Define Support Vector Machines model, preceded by a data standard scaler (subtract mean and divide by std)
        self.model = Pipeline([('scaler', StandardScaler()),
                               ('svm', SVC(C=c,
                                           kernel=kernel,
                                           degree=degree,
                                           gamma=gamma,
                                           random_state=random_state))])

        # Save default parameters
        self.default_params = {'svm__C': c,
                               'svm__kernel': kernel,
                               'svm__degree': degree,
                               'svm__gamma': gamma}

    def ValidationCurve(self, x_train, y_train, **kwargs):
        """Plot model complexity curves with cross-validation.

            Args:
               x_train (ndarray): training data.
               y_train (ndarray): training labels.
               kwargs (dict): additional arguments to pass for model complexity curves plotting:
                    - C_range (ndarray or list): array or list of values for the regularization term.
                    - kernels (ndarray or list): array or list of values for kernels.
                    - gamma_range (ndarray or list): array or list of values for the gamma value.
                    - poly_degrees (ndarray or list): array or list of values for the polynomial degree p.
                    - cv (int): number of k-folds in cross-validation.
                    - ymax (float): lower y axis limit.

            Returns:
               None.
            """
        # Initially our optimal parameters are simply the default parameters
        print('\n\nModel Complexity Analysis')
        self.optimal_params = self.default_params.copy()

        # Initialize best values
        best_score, best_c, best_kernel, best_d, best_gamma = 0., 1., '', 3, 0.001

        # Create a new figure for the regularization term validation curve and set proper arguments
        plt.figure()
        plt.grid()
        kwargs['param'] = 'svm__C'
        kwargs['range'] = kwargs['C_range']
        kwargs['xlabel'] = 'C'
        kwargs['xscale'] = 'linear'

        # For all different kernels
        for kernel in kwargs['kernels']:

            # Set current kernel as if was an optimal parameter
            self.optimal_params['svm__kernel'] = kernel

            kernelname = 'polynomial' if kernel == kwargs['kernels'][0] else 'Radial Basis Function'

            # Set training and validation label for current kernel
            kwargs['label_training'] = 'Training- {} '.format(kernelname)
            kwargs['label_validation'] = 'Validation- {} '.format(kernelname)

            # Plot validation curve for the regualrization term and kernels and get optimal value and score
            c, score = super().ValidationCurve(x_train, y_train, **kwargs)
            print('--> c = {}, kernel = {} --> score = {:.4f}'.format(c, kernel, score))

            # If this score is higher than the best score found so far, update best values
            if score > best_score:
                best_score, best_c, best_kernel = score, c, kernel

        # Save the optimal regularization term and kernel in our dictionary of optimal parameters and save figure
        self.optimal_params['svm__C'] = best_c
        self.optimal_params['svm__kernel'] = best_kernel
        plt.savefig(IMAGE_DIR + '{}_c'.format(self.modelName))
        
        # Set optimal parameters as model parameters
        print('\nBest params have been found to be :\n {}'.format(self.optimal_params))
        self.model.set_params(**self.optimal_params)
