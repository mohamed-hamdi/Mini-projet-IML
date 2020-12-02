#!/usr/bin/env python
"""
Functions used in a machine learning workflow
implemented for a project at IMT Atlantique
(Brest ,FRANCE)
"""

__author__ = "HAMDI Mohamed, KLEIMAN Ilan, OUALI Maher, RAJHI Mohamed"

# imports
import pandas as pd
import numpy as np
import itertools
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
import random
import numpy as np
import numpy.ma as ma
import numbers
from scipy import sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

def models_training_using_pipeline(preprocessor,models,X_train,y_train,X_valid,y_valid):
    """
    Train the models and print accuracy score of each model
    PS : models must be a list of tuples
    @author : Mohamed RAJHI
    """
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model',model)
                                ])
        my_pipeline.fit(X_train, y_train)
        y_pred=my_pipeline.predict(X_valid)
        print("accuracy score for model : {} is {}".format(name,accuracy_score(y_valid,y_pred)))
    return

g

def PlotCorrMatrix(data, FigSize=(10, 10)):
    """Plots the correlation matrix .

    Arguments and function description

    @author : HAMDI Mohamed
    """
    data = pd.DataFrame(data)
    corrMatrix = data.corr()
    plt.figure(figsize=FigSize)
    mask = np.triu(np.ones_like(corrMatrix, dtype=bool))
    cmap = sn.diverging_palette(230, 20, as_cmap=True)
    sn.heatmap(abs(corrMatrix), annot=True, cmap=cmap, mask=mask)
    plt.show()
    return







class CategoricalTransformer(BaseEstimator, TransformerMixin):
    """Returns data with categorical features handled following a strategy given as argument.

    Arguments : strategy = Defines the strategy used to encode categorical features.Possible values :
                           "one_hot_encoding", "ordinal_encoding", "target_encoding".
                categorical_cols_names = Defines the categorical features that will be encoded.
                                         If this argument is not given, the categorical features will
                                         be detected automatically .

    @author : HAMDI Mohamed
    """

    # Class constructor method that takes in a list of values as its argument
    def __init__(self, strategy, categorical_cols_names=None):
        self._replacement_dict = {}
        self._strategy = strategy
        self._categorical_vars = categorical_cols_names

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Transformer method we wrote for this transformer
    def transform(self, X, y=None):
        X = pd.DataFrame(X)
        if self._categorical_vars is None:
            self._categorical_vars = list(X.select_dtypes(include=['string']).columns)
        for col in self._categorical_vars:
            categories = list(X[col].drop_duplicates())
            if self._strategy == "one_hot_encoding":
                aux = pd.get_dummies(X[col], prefix=col)
                X = pd.concat([X, aux], axis=1)
                X = X.drop([col], axis=1)
            if self._strategy == "ordinal_encoding":
                replacement_dic = {categories[i]: i for i in range(len(categories))}
                X[col] = X[col].replace(replacement_dic)
                self._replacement_dict[col] = replacement_dic
            if self._strategy == "target_encoding":
                aux = pd.DataFrame({'col': X[col], 'target': y})
                aux = aux.groupby('col').mean(numeric_only=False)
                replacement_dic = dict(zip(aux.index, aux.target))
                X[col] = X[col].replace(replacement_dic)
                self._replacement_dict[col] = replacement_dic
        return X.values



"""Implemented by Ilan Kleiman"""
"""Class: generates indexes for separating data into train test set. Implemented as iterator """
class TrainTestGenerator:

    def __init__(self, n_splits,test_size=None,train_size=None,data=None,data_length=None):

        """Constructor function for index generator
            Parameters:
            test_size (int,float): (equivalent to train_size)size of the partition represented as total size or a fraction of the data_length (float)
            data_length: Number of elements of the dataset to be partitioned, can be leaved to None if data field is not None
            data: data ND matrix from where the number of data elements to be indexed can be calculated

            Returns:
            Tuple(train, test): Indexes to partition data into a training and testing group for model training and verification.

           """

        self.data_length = data_length
        self.test_size = test_size
        self.train_size = train_size
        self.num = 0
        self.n_splits = n_splits
        self.data = data
        if data is not None:
                self.data_length = data.shape[0]

    def generateTestTrainindexes(self):

        if self.test_size is not None:
            if isinstance(self.test_size, float):
                test_length = int(round(self.test_size * self.data_length))
            else:
                test_length = self.test_size
            train_length = self.data_length - test_length

        if self.train_size is not None:
            if isinstance(self.train_size, float):
                train_length = int(round(self.train_size * self.data_length))
            else:
                train_length = self.train_size

        indexes = np.array((range(0, self.data_length)))

        random.shuffle(indexes)
        train_indexes = indexes[0:train_length]
        test_indexes = indexes[train_length:]

        return train_indexes, test_indexes
    def __iter__(self):
        return self

    def __next__(self):
        if self.num!=self.n_splits:
            self.num += 1
            return self.generateTestTrainindexes()

        else:
            raise StopIteration

    def split(self):
        return [self.generateTestTrainindexes() for i in range(self.n_splits)]

"""Implemented by Ilan Kleiman"""
"""Class in charge of standaraizing data and extraction principal components using PCA"""
class PreProcessor:
    def __init__(self,NbComponents=0,retained_variance=0.0):
        self.NbComponents=NbComponents
        self.retained_variance = retained_variance
        self.scaler=StandardScaler()

        if self.retained_variance == 0.0 and self.NbComponents == 0:
            raise Exception("Wrong configuration of parameters")
        if self.retained_variance != 0.0:
            self.pca = PCA(self.retained_variance)
        elif self.NbComponents != 0:
            self.pca = PCA(n_components=self.NbComponents)

    def fit(self,data):
        # Standardizing the features
        temp_values = self.scaler.fit_transform(data)
        self.pca.fit(temp_values)

    def transform(self,data):
        temp_values = self.scaler.transform(data)
        temp_values = self.pca.transform(temp_values)
        return temp_values

    def fit_transform(self,data):
        self.fit(data)
        return self.transform(data)

class GridSearchHyperParamsCV:
    """Returns the best hyperparameters for the model and it performance after a grid search .

    Arguments:  model = model to use .
                cv_splitter = cross-validation generator , splits the data to train and validation sets
                              by giving the indexes at each iteration.We can use our custom cross-validation "CustomShuffle"
                              or any other cross-validation generator from sklearn like KFold
                parameters = Dictionary with parameters names as keys and lists of parameter settings to try as values for our model.
                n_jobs =  Number of jobs to run in parallel. -1 means using all processors.
                verbose = Controls the verbosity: the higher, the more messages.

    @author : HAMDI Mohamed & OUALI Maher
    """

    # Class constructor method that takes in a list of values as its argument
    def __init__(self, model, cv_splitter, parameters, n_jobs, verbose):
        self._model = model
        self._cv_splitter = cv_splitter
        self._parameters = parameters
        self._n_jobs = n_jobs
        self._verbose = verbose
        self._scores = None
        self._best_params = None
        self._best_score = None
        self._grid = None

    def create_grid(self):
        labels, terms = zip(*self._parameters.items())
        self._grid = [dict(zip(labels, term)) for term in itertools.product(*terms)]

    def model_fit_score(self, Xt, yt, Xv, yv, params):
        self._model.set_params(**params)
        self._model.fit(Xt, yt)
        return self._model.score(Xv, yv)

    def fit(self, X, y):
        self.create_grid()
        results = []
        for train, validation in self._cv_splitter.split(X):
            r = Parallel(n_jobs=self._n_jobs, verbose=self._verbose)(
                delayed(self.model_fit_score)(X[train], y[train], X[validation], y[validation], params) for params in
                self._grid)
            results.append(list(r))
        self._scores = np.mean(np.array(results), axis=0)
        self._best_score = np.max(self._scores)
        self._best_params = self._grid[np.argmax(self._scores)]


class FeatureSelectorCorr(BaseEstimator, TransformerMixin):
    """Returns the data without correlated features based on a threshold for correlation given as argument
        if k features have pairwise correlation greater than threshold only one is kept .

    Arguments : threshold = Defines the threshold for correlation.
                method = Defines the method of correlation . Possible values are {‘pearson’, ‘kendall’, ‘spearman’}.
                default value is 'pearson'.

    @author : HAMDI Mohamed
    """

    # Class constructor method that takes in a list of values as its argument
    def __init__(self, threshold, method='pearson'):
        self._threshold = threshold
        self._method = method
        self._dropped_features = None

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Transformer method we wrote for this transformer
    def transform(self, X, y=None):
        X = pd.DataFrame(X).convert_dtypes()
        corrMatrix = X.corr(method=self._method)
        corrMatrix = corrMatrix.unstack()
        top_corr = list(corrMatrix[corrMatrix >= self._threshold].index)

        to_drop = []
        to_keep = list(X.columns)
        for e in top_corr:
            print(e)
            if (e[0] in to_keep) and (e[1] in to_keep) and (e[0] != e[1]):
                to_drop.append(e[1])
                to_keep.remove(e[1])
        self._dropped_features = to_drop
        X = X.drop(to_drop, axis=1)
        return X.values


class CustomImputer(BaseEstimator, TransformerMixin):
    """Imputes missing values using a given strategy

    @author : RAJHI Mohamed
    """
    def __init__(self,strategy="mean",missing_values=np.nan):
        self.strategy=strategy
        self.missing_values=missing_values
    
    
    #function to fit our data
    def fit(self, X, y=None):
        return self
    
   
    #function to transform our data
    def transform(self, X):
                
                # X = pd.DataFrame(X).convert_dtypes()
                # Mean
                import numpy as np
                if self.strategy == "mean":
                    X=X.apply(lambda x:x.fillna(x.mean()))
                # Most frequent
                elif self.strategy == "most_frequent":
                    X=X.apply(lambda x:x.fillna(x.value_counts().index[0]))
                if self.strategy == "median":
                    X=X.apply(lambda x:x.fillna(x.median()))
                

               
                return X.values
            
 

