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
