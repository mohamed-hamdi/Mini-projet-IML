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
from sklearn.exceptions import NotFittedError


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
        else
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
        else
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
        else
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
        else
            print("This Normalizer has been fitted over some data having the following statistics \n\t*Maxes: {}".format(self.max.tolist()))
        

class RobustScaler():
    """ This function applies Robust scaling over a given dataset
     
        Arguments: data = data to be scaled (or reverse scaled)
                   quantile_range = the percentage of data (upper, lower) to be used for scaling the data
        
        @author:  OUALI Maher
    """
    def __init__(self, quantile_range=(25, 75):tuple):
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
    
