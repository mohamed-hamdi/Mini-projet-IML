#!/usr/bin/env python
"""
Functions used in a machine learning workflow
implemented for a project at IMT Atlantique
(Brest ,FRANCE)
"""

__author__ = "HAMDI Mohamed, KLEIMAN Ilan, OUALI Maher, RAJHI Mohamed"

# imports
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import random
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def function():
    """Returns description .

    Arguments and function description

    @author : Author of the function
    """
    # Code here

    return


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    """Returns data with categorical features handled following a strategy given as argument.

    Arguments and function description

    @author : HAMDI Mohamed
    """

    # Class constructor method that takes in a list of values as its argument
    def __init__(self, strategy, categorical_cols_names=None):
        self._strategy = strategy
        self._categorical_vars = categorical_cols_names

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Transformer method we wrote for this transformer
    def transform(self, X, y=None):

        X = pd.DataFrame(X).convert_dtypes()
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
            if self._strategy == "target_encoding":
                aux = pd.DataFrame({'col': X[col], 'target': y})
                aux = aux.groupby('col').mean(numeric_only=False)
                replacement_dic = dict(zip(aux.index, aux.target))
                X[col] = X[col].replace(replacement_dic)
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
