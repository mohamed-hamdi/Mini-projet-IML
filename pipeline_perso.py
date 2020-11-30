#!/usr/bin/env python
"""
Functions used in a machine learning workflow
implemented for a project at IMT Atlantique
(Brest ,FRANCE)
"""

__author__ = "HAMDI Mohamed, KLEIMAN Ilan, OUALI Maher, RAJHI Mohamed"

# imports
import pandas as pd
import itertools
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin , clone
from sklearn.exceptions import NotFittedError


import random
import numpy as np
import numpy.ma as ma
import numbers
import warnings

from sklearn.decomposition import PCA
from sklearn.impute._base import _most_frequent
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed


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

def is_scalar_nan(x):
    
    """is_scalar_nan(x) is True does not fail.

    @author : RAJHI Mohamed
    """
    return bool(isinstance(x, numbers.Real) and np.isnan(x))

def _object_dtype_isnan(X):
    """@author : RAJHI Mohamed"""
    return X != X

def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == value_to_mask.
       @author : RAJHI Mohamed
    """
    if is_scalar_nan(value_to_mask):
        if X.dtype.kind == "f":
            return np.isnan(X)
        elif X.dtype.kind in ("i", "u"):
            # can't have NaNs in integer array.
            return np.zeros(X.shape, dtype=bool)
        else:
            # np.isnan does not work on object dtypes.
            return _object_dtype_isnan(X)
    else:
        return X == value_to_mask





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

    def __init__(self, n_splits, test_size=None, train_size=None, data=None, data_length=None):

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
        if self.num != self.n_splits:
            self.num += 1
            return self.generateTestTrainindexes()

        else:
            raise StopIteration

    def split(self):
        return [self.generateTestTrainindexes() for i in range(self.n_splits)]


"""Implemented by Ilan Kleiman"""
"""Class in charge of standaraizing data and extraction principal components using PCA"""


class PreProcessor:
    def __init__(self, NbComponents=0, retained_variance=0.0):
        self.NbComponents = NbComponents
        self.retained_variance = retained_variance
        self.scaler = StandardScaler()

        if self.retained_variance == 0.0 and self.NbComponents == 0:
            raise Exception("Wrong configuration of parameters")
        if self.retained_variance != 0.0:
            self.pca = PCA(self.retained_variance)
        elif self.NbComponents != 0:
            self.pca = PCA(n_components=self.NbComponents)

    def fit(self, data):
        # Standardizing the features
        temp_values = self.scaler.fit_transform(data)
        self.pca.fit(temp_values)

    def transform(self, data):
        temp_values = self.scaler.transform(data)
        temp_values = self.pca.transform(temp_values)
        return temp_values

    def fit_transform(self, data):
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
        self._model = clone(self._model)
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
        # retrain model with the best parameters .
        self._model = clone(self._model)
        self._model.set_params(**self._best_params)
        self._model.fit(X, y)

    def predict(self, X_test):
        if(self._best_params != None):
            return self._model.predict(X_test)
        else:
            raise NotFittedError("GridSearch for Hyper parameters tuning has not been called yet. call 'fit' method")

    def score(self, X_test, y_test):
        if(self._best_params != None):
            return self._model.score(X_test,y_test)
        else:
            raise NotFittedError("GridSearch for Hyper parameters tuning has not been called yet. call 'fit' method")



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
    
    
    
    def fit(self, X, y=None):
        return self
    
   
    
    def transform(self, X):
               
                X=np.array(X)
                mask = _get_mask(X, self.missing_values)
                masked_X = ma.masked_array(X, mask=mask)

                # Mean
                if self.strategy == "mean":
                    mean_masked = np.ma.mean(masked_X, axis=0)
                    mean = np.ma.getdata(mean_masked)
                    mean[np.ma.getmask(mean_masked)] = np.nan
                    statistics =mean
                # Most frequent
                elif self.strategy == "most_frequent":
                    X = X.transpose()
                    mask = mask.transpose()

                    if X.dtype.kind == "O":
                        most_frequent = np.empty(X.shape[0], dtype=object)
                    else:
                        most_frequent = np.empty(X.shape[0])

                    for i, (row, row_mask) in enumerate(zip(X[:], mask[:])):
                        row_mask = np.logical_not(row_mask).astype(np.bool)
                        row = row[row_mask]
                        most_frequent[i] = _most_frequent(row, np.nan, 0)
                    statistics = most_frequent
                
                #handling invalid mask
                invalid_mask = _get_mask(statistics, np.nan)
                valid_mask = np.logical_not(invalid_mask)
                valid_statistics = statistics[valid_mask]
                valid_statistics_indexes = np.flatnonzero(valid_mask)

                if invalid_mask.any():
                    missing = np.arange(X.shape[1])[invalid_mask]
                    if self.verbose:
                        warnings.warn("Deleting features without "
                                      "observed values: %s" % missing)
                    X = X[:, valid_statistics_indexes]

                # Do actual imputation
          
                mask = _get_mask(X, self.missing_values)
                n_missing = np.sum(mask, axis=0)
                values = np.repeat(valid_statistics, n_missing)
                coordinates = np.where(mask.transpose())[::-1]

                X[coordinates] = values
                return X
            
 

class CustomScaler():
    """
           This class applies one of the scaler defined bellow .

           Arguments: data = data to be scaled (or reverse scaled)
                      Scaler= scaler name to be used. Default value "StandardScaler"



           @author:  OUALI Maher
       """

    def __init__(self, scaler_name="StandardScaler"):
        self.scaler_name = scaler_name
        if self.scaler_name == "StandardScaler":
            self.scaler = StandardScaler()
        elif self.scaler_name == "MaxAbsScaler":
            self.scaler = MaxAbsScaler()
        elif self.scaler_name == "MinMaxScaler":
            self.scaler = MinMaxScaler()
        elif self.scaler_name == "Normalizer":
            self.scaler = Normalizer()
        elif self.scaler_name == "RobustScaler":
            self.scaler = RobustScaler()
        else:
            raise TypeError("Unkown scaler : possible scaler names :{ 'StandardScaler' ,'MaxAbsScaler','MinMaxScaler','Normalizer','RobustScaler'}")

    def fit(self, data):
        return self.scaler.fit(data)

    def transform(self, data):
        return self.scaler.transform(data)

    def fit_transform(self, data):
        return self.scaler.fit_transform(data)

    def reverse_transform(self, data):
        return self.scaler.reverse_transform(data)

class StandardScaler():
    """
        This class applies standard scaling over a dataset, for each sample x it calculates z = (x - m)/s
        Where m is the mean of the given dataset and s its standard deviation
        
        Arguments: data = data to be scaled (or reverse scaled)
        
        
        @author:  OUALI Maher
    """
    def __init__(self):
        self.m = None
        self.s = None

    def fit(self, data):
        """ This function calculates all means and stds for different features and therefore fitting the scaler on the given data """
        self.m = np.mean(data, axis=0)
        self.s = np.std(data, axis=0)
        return self

    def transform(self, data):
        """ This function scales data samples with correspondance to a list of means and stds """
        #check if scaler has been fitted or not
        if(self.m == None or self.s == None):
            raise NotFittedError("Standard Scaler not fitted yet. Call either 'fit' or 'fit_transform' in order to fit data")
        #check if data has the same shape as the parameters
        if(len(self.m) != data.shape[1]):
            raise ValueError("Data number of columns '{}' and parameters length '{}' doesn't match".format(data.shape[1], len(self.m)))
        #apply direct transformation or raise value error
        result = np.zeros(data.shape)
        for i in range(data.shape[1]):
            if(self.s[i] != 0):
                result[:,i] = (data[:,i] - self.m[i]) / self.s[i]
            else:
                result[:,i] = data[:,i] - self.m[i]
        return result

    def fit_transform(self, data):
        """ This function applies the fitting and transformation at the same time """
        self.fit(self, data)
        return self.transform(self, data)

    def reverse_transform(self, data):
        #check if scaler has been fitted or not
        if(self.m == None or self.s == None):
            raise NotFittedError("Standard Scaler not fitted yet. Call either 'fit' or 'fit_transform' in order to fit data")
        #check if data has the same shape as the parameters
        if(len(self.m) != data.shape[1]):
            raise ValueError("Data number of columns '{}' and parameters length '{}' doesn't match".format(data.shape[1], len(self.m)))
        #apply reverse transformation or raise value error
        result = np.zeros(data.shape)
        for i in range(data.shape[1]):     
            result[:,i] = data[:,i] * self.s[i] + self.m[i]
        return result

    def describe(self):
        if(self.min == None or self.max == None):
            print("This Standard Scaler has not been fitted yet")
        else :
            print("This Standard Scaler has been fitted over some data having the following statistics \n\t*Means : {} \n\t*Stds: {}".format(self.m.tolist(), self.s.tolist()))
        


class MaxAbsScaler():
    """
        This class applies MaxAbs scaling over a dataset, for each sample x it calculates z = x / maxabs
        Where max is the maximum of all absolute values of a given dataset
        
         
        Arguments: data = data to be scaled (or reverse scaled)
        
        
        @author:  OUALI Maher
    """
    def __init__(self):
        self.maxabs = None

    def fit(self, data):
        """ This function calculates all maximum absolute values for different features and therefore fitting the scaler on the given data """
        self.maxabs = np.max(np.absolute(data), axis=0)
        return self

    def transform(self, data) -> np.array:
        """ This function scales data samples with correspondance to a list of mins and maxes """
        #check if scaler has been fitted or not
        if(self.maxabs == None):
            raise NotFittedError("MaxAbs Scaler not fitted yet. Call either 'fit' or 'fit_transform' in order to fit data")
        #check if data has the same shape as the parameters
        if(len(self.maxabs) != data.shape[1]):
            raise ValueError("Data number of columns '{}' and parameters length '{}' doesn't match".format(data.shape[1], len(self.maxabs)))
        #apply direct transformation or raise value error
        result = np.zeros(data.shape)
        for i in range(data.shape[1]):
            if(self.maxabs[i] != 0):
                result[:,i] = data[:,i] / self.maxabs[i]
        return result

    def fit_transform(self, data) -> np.array:
        """ This function applies the fitting and transformation at the same time """
        self.fit(self, data)
        return self.transform(self, data)

    def reverse_transform(self, data) -> np.array:
        #check if scaler has been fitted or not
        if(self.maxabs == None):
            raise NotFittedError("MaxAbs Scaler not fitted yet. Call either 'fit' or 'fit_transform' in order to fit data")
        #check if data has the same shape as the parameters
        if(len(self.maxabs) != data.shape[1]):
            raise ValueError("Data number of columns '{}' and parameters length '{}' doesn't match".format(data.shape[1], len(self.maxabs)))
        #apply reverse transformation or raise value error
        result = np.zeros(data.shape)
        for i in range(data.shape[1]):
            result[:,i] = data[:,i] * self.maxabs[i]
        return result

    def describe(self):
        if(self.maxabs == None):
            print("This MaxAbs Scaler has not been fitted yet")
        else :
            print("This MaxAbs Scaler has been fitted over some data having the following statistics \n\t*Absolute Maxes : {}".format(self.maxabs.tolist()))


class MinMaxScaler():
    """
        This class applies minmax scaling over a dataset, for each sample x it calculates z = (x - min)/(min + max)
        Where min is the minimum of the given dataset and max its maximum
        
        Arguments: data = data to be scaled (or reverse scaled)
        
        
        @author:  OUALI Maher
    """
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data):
        """ This function calculates all minimum and maximum values for different features and therefore fitting the scaler on the given data """
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        return self

    def transform(self, data) -> np.array:
        """ This function scales data samples with correspondance to a list of mins and maxes """
        #check if scaler has been fitted or not
        if(self.min == None or self.max == None):
            raise NotFittedError("MinMax Scaler not fitted yet. Call either 'fit' or 'fit_transform' in order to fit data")
        #check if data has the same shape as the parameters
        if(len(self.max) != data.shape[1]):
            raise ValueError("Data number of columns '{}' and parameters length '{}' doesn't match".format(data.shape[1], len(self.max)))
        #apply direct transformation or raise value error
        result = np.zeros(data.shape)
        for i in range(data.shape[1]):
            if((self.min[i] + self.max[i]) != 0):
                result[:,i] = (data[:,i] - self.min[i]) / (self.min[i] + self.max[i])
            else:
                result[:,i] = data[:,i] - self.min[i]
        return result

    def fit_transform(self, data) -> np.array:
        """ This function applies the fitting and transformation at the same time """
        self.fit(self, data)
        return self.transform(self, data)

    def reverse_transform(self, data) -> np.array:
        #check if scaler has been fitted or not
        if(self.min == None or self.max == None):
            raise NotFittedError("MinMax Scaler not fitted yet. Call either 'fit' or 'fit_transform' in order to fit data")
        #check if data has the same shape as the parameters
        if(len(self.max) != data.shape[1]):
            raise ValueError("Data number of columns '{}' and parameters length '{}' doesn't match".format(data.shape[1], len(self.max)))
        #apply reverse transformation or raise value error
        result = np.zeros(data.shape)
        for i in range(data.shape[1]):
            result[:,i] = data[:,i] * (self.min[i] + self.max[i]) + self.min[i]
        return result

    def describe(self):
        if(self.min == None or self.max == None):
            print("This MinMax Scaler has not been fitted yet")
        else :
            print("This MinMax Scaler has been fitted over some data having the following statistics \n\t*Mins : {} \n\t*Maxes: {}".format(self.min.tolist(), self.max.tolist()))


class Normalizer():
    """This class applies normalization over a dataset
    
    
     
        Arguments: data = data to be scaled (or reverse scaled)
                   norm = the norm to be used in the normalization, the accepted norms are 'l2' or 'l1': for 'l2' norm the sum of the square of all scaled samples (over a row)
                   is equal to 1 where for the 'l1' norm the sum of the absolute value is equal to 1
        
        
        @author:  OUALI Maher
    """
    def __init__(self, norm='l2'):
        if(norm not in ['l2', 'l1']):
            raise ValueError("norm must be in ['l2', 'l1']")
        self.norm = norm
        self.max = None

    def fit(self, data):
        """ This function fits the scaler over the data by determining the maximum values and the maximum absolute value for each row """
        self.max = np.max(data, axis=1)
        self.maxabs = np.max(np.absolute(data), axis=1)
        return self

    def transform(self, data):
        #check if scaler has been fitted or not
        if(self.max == None):
            raise NotFittedError("Normalizer not fitted yet. Call either 'fit' or 'fit_transform' in order to fit data")
        #check if data has the same shape as the parameters
        if(len(self.max) != data.shape[0]):
            raise ValueError("Data number of rows '{}' and parameters length '{}' doesn't match".format(data.shape[0], len(self.max)))
        #apply direct transformation or raise value error
        result = np.zeros(data.shape)
        if(self.norm == 'l2'):
            for i in range(data.shape[1]):
                if(self.maxabs[i] != 0):
                    result[i,:] = (data[i,:] / self.max[i]) * np.sqrt(1/np.sum(np.square((data[i,:] / self.max[i]))))
            return result
        else:
            for i in range(data.shape[1]):
                if(self.maxabs[i] != 0):
                    result[i,:] = (data[i,:] / self.max[i]) * (1/np.sum(np.absolute((data[i,:] / self.max[i]))))
            return result

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def reverse_transform(self, data):
        #check if scaler has been fitted or not
        if(self.max == None):
            raise NotFittedError("Normalizer not fitted yet. Call either 'fit' or 'fit_transform' in order to fit data")
        #check if data has the same shape as the parameters
        if(len(self.max) != data.shape[0]):
            raise ValueError("Data number of rows '{}' and parameters length '{}' doesn't match".format(data.shape[0], len(self.max)))
        #apply reverse transformation or raise value error
        result = np.zeros(data.shape)
        for i in range(data.shape[1]):
            if(self.maxabs[i] != 0):
                result[i,:] = (data[i,:] / self.max[i]) * np.sqrt(1/np.sum(np.square((data[i,:] / self.max[i]))))
        return result

    def describe(self):
        if(self.max == None):
            print("This Normalizer has not been fitted yet")
        else :
            print("This Normalizer has been fitted over some data having the following statistics \n\t*Maxes: {}".format(self.max.tolist()))
        

class RobustScaler():
    """ This function applies Robust scaling over a given dataset
     
        Arguments: data = data to be scaled (or reverse scaled)
                   quantile_range = the percentage of data (upper, lower) to be used for scaling the data
        
        @author:  OUALI Maher
    """
    def __init__(self, quantile_range=(25, 75)):
        if(quantile_range[0] >= quantile_range[1]):
            raise ValueError("first value of tuple must be stricly smaller than the second tuple value")
        if(quantile_range[0] <=0 or quantile_range[1] >= 100):
            raise ValueError("first tuple value must be > 0 and second tuple value must be < 100")
        self.quantile_range = quantile_range
        self.median = None
        self.upper_quantile = None
        self.lower_quantile = None

    def fit(self, data:np.array):
        """ This functions fits the scaler over data by determining median values and upper (lower) quantiles in different columns """
        self.median = np.median(data, axis=0)
        self.upper_quantile = np.percentile(data, q=self.quantile_range[1], axis=0)
        self.lower_quantile = np.percentile(data, q=self.quantile_range[0], axis=0)
        return self

    def transform(self, data:np.array) -> np.array:
        #check if scaler has been fitted or not
        if(self.median == None or self.upper_quantile == None or self.lower_quantile == None):
            raise NotFittedError("Robust Scaler not fitted yet. Call either 'fit' or 'fit_transform' in order to fit data")
        #check if data has the same shape as the parameters
        if(len(self.median) != data.shape[1]):
            raise ValueError("Data number of columns '{}' and parameters length '{}' doesn't match".format(data.shape[1], len(self.median)))
        #apply direct transformation or raise value error
        result = np.zeros(data.shape)
        for i in range(data.shape[1]):
            if((self.upper_quantile[i] + self.lower_quantile[i]) != 0):
                result[:,i] = (data[:,i] - self.median[i] - self.lower_quantile[i]) / (self.upper_quantile[i] + self.lower_quantile[i])
            else:
                result[:,i] = (data[:,i] - self.median[i] - self.lower_quantile[i])
        return result

    def fit_transform(self, data:np.array) -> np.array:
        self.fit(data)
        return self.transform(data)

    def reverse_transform(self, data:np.array) -> np.array:
        #check if scaler has been fitted or not
        if(self.median == None or self.upper_quantile == None or self.lower_quantile == None):
            raise NotFittedError("Robust Scaler not fitted yet. Call either 'fit' or 'fit_transform' in order to fit data")
        #check if data has the same shape as the parameters
        if(len(self.median) != data.shape[1]):
            raise ValueError("Data number of columns '{}' and parameters length '{}' doesn't match".format(data.shape[1], len(self.median)))
        #apply reverse transformation or raise value error
        result = np.zeros(data.shape)
        for i in range(data.shape[1]):
            result[:,i] = data[:,i] * (self.upper_quantile[i] + self.lower_quantile[i]) + self.median[i] + self.lower_quantile[i]
        return result