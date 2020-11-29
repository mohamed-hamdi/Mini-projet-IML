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
import numpy.ma as ma
import numbers
import numpy as np 
def function():
    """Returns description .

    Arguments and function description

    @author : Author of the function
    """
    # Code here

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
            
 
 