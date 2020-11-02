# Script to run experiments

import numpy as np

from clustering import KMeansClustering, MixtureOfGaussians
from dimensionality_reduction import IndependentComponents, KernelPrincipalComponents, \
                                     PrincipalComponents, RandomProjections
from neural_networks import NeuralNetwork


from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd

IMAGE_DIR = 'images/'  # output images directory


def load_dataset(dataset='WDBC', split_percentage=0.3):


    # Load WDBC
    data = load_breast_cancer()
    x, y, labels, features = data.data, data.target, data.target_names, data.feature_names


    # Split dataset in training and validation sets, preserving classes representation
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_percentage, shuffle=True,
                                                        random_state=42, stratify=y)

    # Normalize feature data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print('\nTotal dataset size:')
    print('Number of instances: {}'.format(x.shape[0]))
    print('Number of features: {}'.format(x.shape[1]))
    print('Number of classes: {}'.format(len(labels)))
    print('Training Set : {}'.format(x_train.shape))
    print('Testing Set : {}'.format(x_test.shape))

    return x_train, x_test, y_train, y_test


def clustering(x_train, x_test, y_train, **kwargs):

    # Declare kMeans, perform experiments and get clusters on training data
    kmeans = KMeansClustering(n_clusters=kwargs['kmeans_n_clusters'], max_n_clusters=8)
    kmeans_clusters = kmeans.experiment(x_train, x_test, y_train,
                                        dataset=kwargs['dataset'],
                                        perform_model_complexity=kwargs['perform_model_complexity'])

    print('\n--------------------------')
    print('GMM')

    # Declare Gaussian Mixtures Models, perform experiments and get clusters on training data
    gmm = MixtureOfGaussians(n_clusters=kwargs['em_n_clusters'], covariance=kwargs['em_covariance'], max_n_clusters=8)
    gmm_clusters = gmm.experiment(x_train, x_test, y_train,
                                  dataset=kwargs['dataset'],
                                  perform_model_complexity=kwargs['perform_model_complexity'])

    return kmeans_clusters, gmm_clusters


def dimensionality_reduction(x_train, x_test, y_train, **kwargs):

    # Declare PCA, perform experiments by reducing the dataset and perform clustering experiments on it
    pca = PrincipalComponents(n_components=kwargs['pca_n_components'])
    x_pca = pca.experiment(x_train, x_test, y_train,
                           dataset=kwargs['dataset'],
                           perform_model_complexity=kwargs['perform_model_complexity'])

    clustering(x_pca[0],  x_pca[1], y_train,
               dataset=kwargs['dataset'] + '_pca_reduced',
               kmeans_n_clusters=kwargs['pca_kmeans_n_clusters'],
               em_n_clusters=kwargs['pca_em_n_clusters'], em_covariance=kwargs['pca_em_covariance'],
               perform_model_complexity=kwargs['perform_model_complexity'])

    print('\n--------------------------')
    print('ICA')
    print('--------------------------')

    # Declare ICA, perform experiments by reducing the dataset and perform clustering experiments on it
    ica = IndependentComponents(n_components=kwargs['ica_n_components'])
    x_ica = ica.experiment(x_train, x_test, y_train,
                           dataset=kwargs['dataset'],
                           perform_model_complexity=kwargs['perform_model_complexity'])

    clustering(x_ica[0],  x_ica[1], y_train,
               dataset=kwargs['dataset'] + '_ica_reduced',
               kmeans_n_clusters=kwargs['ica_kmeans_n_clusters'],
               em_n_clusters=kwargs['ica_em_n_clusters'], em_covariance=kwargs['ica_em_covariance'],
               perform_model_complexity=kwargs['perform_model_complexity'])

    print('\n--------------------------')
    print('KPCA')
    print('--------------------------')

    # Declare KPCA, perform experiments by reducing the dataset and perform clustering experiments on it
    kpca = KernelPrincipalComponents(n_components=kwargs['kpca_n_components'], kernel=kwargs['kpca_kernel'])
    x_kpca = kpca.experiment(x_train, x_test, y_train,
                             dataset=kwargs['dataset'],
                             perform_model_complexity=kwargs['perform_model_complexity'])

    clustering(x_kpca[0], x_kpca[1], y_train,
               dataset=kwargs['dataset'] + '_kpca_reduced',
               kmeans_n_clusters=kwargs['kpca_kmeans_n_clusters'],
               em_n_clusters=kwargs['kpca_em_n_clusters'], em_covariance=kwargs['kpca_em_covariance'],
               perform_model_complexity=kwargs['perform_model_complexity'])

    print('\n--------------------------')
    print('RP')
    print('--------------------------')

    # Declare RP, perform experiments by reducing the dataset and perform clustering experiments on it
    rp = RandomProjections(n_components=kwargs['rp_n_components'])
    x_rp = rp.experiment(x_train, x_test, y_train,
                         dataset=kwargs['dataset'],
                         perform_model_complexity=kwargs['perform_model_complexity'])

    clustering(x_rp[0], x_rp[1], y_train,
               dataset=kwargs['dataset'] + '_rp_reduced',
               kmeans_n_clusters=kwargs['rp_kmeans_n_clusters'],
               em_n_clusters=kwargs['rp_em_n_clusters'], em_covariance=kwargs['rp_em_covariance'],
               perform_model_complexity=kwargs['perform_model_complexity'])

    return x_pca, x_ica, x_kpca, x_rp


def neural_network(x_train, x_test, y_train, y_test,
                   x_pca, x_ica, x_kpca, x_rp,
                   x_kmeans, x_gmm, **kwargs):

    print('--------------------------')

    # Declare Neural Network and perform experiments on the original dataset
    nn = NeuralNetwork(layer1_nodes=kwargs['layer1_nodes'],
                       layer2_nodes=kwargs['layer2_nodes'],
                       learning_rate=kwargs['learning_rate'])
    nn.experiment(x_train, x_test, y_train, y_test)

    print('\n--------------------------')
    print('PCA + NN')
    print('--------------------------')

    # Declare Neural Network and perform experiments on the reduced dataset by PCA
    nn = NeuralNetwork(layer1_nodes=kwargs['layer1_nodes'],
                       layer2_nodes=kwargs['layer2_nodes'],
                       learning_rate=kwargs['learning_rate'])
    nn.experiment(x_pca[0], x_pca[1], y_train, y_test)

    print('\n----------experiments on the reduced dataset by PCA----------------')
    print('ICA + NN')
    print('--------------------------')

    # Declare Neural Network and perform experiments on the reduced dataset by ICA
    nn = NeuralNetwork(layer1_nodes=kwargs['layer1_nodes'],
                       layer2_nodes=kwargs['layer2_nodes'],
                       learning_rate=kwargs['learning_rate'])
    nn.experiment(x_ica[0], x_ica[1], y_train, y_test)

    print('\n--------------------------')
    print('KPCA + NN')
    print('--------------------------')

    # Declare Neural Network and perform experiments on the reduced dataset by KPCA
    nn = NeuralNetwork(layer1_nodes=kwargs['layer1_nodes'],
                       layer2_nodes=kwargs['layer2_nodes'],
                       learning_rate=kwargs['learning_rate'])
    nn.experiment(x_kpca[0], x_kpca[1], y_train, y_test)

    print('\n--------------------------')
    print('RP+ NN')
    print('--------------------------')

    # Declare Neural Network and perform experiments on the reduced dataset by RP
    nn = NeuralNetwork(layer1_nodes=kwargs['layer1_nodes'],
                       layer2_nodes=kwargs['layer2_nodes'],
                       learning_rate=kwargs['learning_rate'])
    nn.experiment(x_rp[0], x_rp[1], y_train, y_test)

    print('\n----------experiments on Kmeans Augmented dataset----------------')
    print('KMEANS+ NN')
    print('--------------------------')

    # Declare Neural Network
    nn = NeuralNetwork(layer1_nodes=kwargs['layer1_nodes'],
                       layer2_nodes=kwargs['layer2_nodes'],
                       learning_rate=kwargs['learning_rate'])

    # Augment the original dataset by adding clusters produced by k-Means as features
    x_kmeans_normalized = (x_kmeans[0] - np.mean(x_kmeans[0])) / np.std(x_kmeans[0])
    x_kmeans_normalized = np.expand_dims(x_kmeans_normalized, axis=1)
    x_train_new = np.append(x_train, x_kmeans_normalized, axis=1)
    x_kmeans_normalized = (x_kmeans[1] - np.mean(x_kmeans[1])) / np.std(x_kmeans[1])
    x_kmeans_normalized = np.expand_dims(x_kmeans_normalized, axis=1)
    x_test_new = np.append(x_test, x_kmeans_normalized, axis=1)

    # Perform experiments on it
    nn.experiment(x_train_new, x_test_new, y_train, y_test)

    print('\n---------experiments on GMM Augmented dataset-----------------')
    print('GMM+ NN')
    print('--------------------------')

    # Declare Neural Network
    nn = NeuralNetwork(layer1_nodes=kwargs['layer1_nodes'],
                       layer2_nodes=kwargs['layer2_nodes'],
                       learning_rate=kwargs['learning_rate'])

    # Augment the original dataset by adding clusters produced by Gaussian Mixture Models as features
    x_gmm_normalized = (x_gmm[0] - np.mean(x_gmm[0])) / np.std(x_gmm[0])
    x_gmm_normalized = np.expand_dims(x_gmm_normalized, axis=1)
    x_train_new = np.append(x_train, x_gmm_normalized, axis=1)
    x_gmm_normalized = (x_gmm[1] - np.mean(x_gmm[1])) / np.std(x_gmm[1])
    x_gmm_normalized = np.expand_dims(x_gmm_normalized, axis=1)
    x_test_new = np.append(x_test, x_gmm_normalized, axis=1)

    # Perform experiments on it
    nn.experiment(x_train_new, x_test_new, y_train, y_test)


if __name__ == '__main__':

    # # Run experiments on WDBC
    print('\n--------------------------')
    dataset = 'WDBC'
    perform_model_complexity = True
    x_train, x_test, y_train, y_test = load_dataset(dataset)

    # Clustering experiments
    kmeans_clusters, gmm_clusters = clustering(x_train, x_test, y_train,
                                               dataset=dataset,
                                               kmeans_n_clusters=2,
                                               em_n_clusters=2, em_covariance='full',
                                               perform_model_complexity=perform_model_complexity)

    # # # Dimensionality reduction experiments
    x_pca, x_ica, x_kpca, x_rp = dimensionality_reduction(x_train, x_test, y_train,
                                                          dataset=dataset,
                                                          pca_n_components=10, pca_kmeans_n_clusters=2,
                                                          pca_em_n_clusters=2, pca_em_covariance='diag',
                                                          ica_n_components=10, ica_kmeans_n_clusters=2,
                                                          ica_em_n_clusters=2, ica_em_covariance='diag',
                                                          kpca_n_components=10, kpca_kernel='cosine',
                                                          kpca_kmeans_n_clusters=2,
                                                          kpca_em_n_clusters=2, kpca_em_covariance='diag',
                                                          rp_n_components=20, rp_kmeans_n_clusters=2,
                                                          rp_em_n_clusters=2, rp_em_covariance='full',
                                                          perform_model_complexity=perform_model_complexity)

    # Neural Network experiments
    neural_network(x_train, x_test, y_train, y_test,
                   x_pca, x_ica, x_kpca, x_rp,
                   kmeans_clusters, gmm_clusters,
                   layer1_nodes=42, layer2_nodes=28, learning_rate=0.04)

    # #Run experiments on HEART
    dataset = 'HEART'
    data = pd.read_csv(
            'data/heart_failure_clinical_records_dataset.csv', header=None).values
    x, y = data[:, :-1], data[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True,
                                                        random_state=42, stratify=y)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    #Clustering experiments
    kmeans_clusters, gmm_clusters = clustering(x_train, x_test, y_train,
                                               dataset=dataset,
                                               kmeans_n_clusters=8,
                                               em_n_clusters=8, em_covariance='full',
                                               perform_model_complexity=perform_model_complexity)

    # # Dimensionality reduction experiments
    x_pca, x_ica, x_kpca, x_rp = dimensionality_reduction(x_train, x_test, y_train,
                                                          dataset=dataset,
                                                          pca_n_components=10, pca_kmeans_n_clusters=8,
                                                          pca_em_n_clusters=8, pca_em_covariance='full',
                                                          ica_n_components=10, ica_kmeans_n_clusters=8,
                                                          ica_em_n_clusters=8, ica_em_covariance='full',
                                                          kpca_n_components=10, kpca_kernel='cosine',
                                                          kpca_kmeans_n_clusters=8,
                                                          kpca_em_n_clusters=3, kpca_em_covariance='full',
                                                          rp_n_components=10, rp_kmeans_n_clusters=4,
                                                          rp_em_n_clusters=4, rp_em_covariance='full',
                                                          perform_model_complexity=perform_model_complexity)

    # Neural Network experiments
    neural_network(x_train, x_test, y_train, y_test,
                   x_pca, x_ica, x_kpca, x_rp,
                   kmeans_clusters, gmm_clusters,
                   layer1_nodes=50, layer2_nodes=30, learning_rate=0.1)
