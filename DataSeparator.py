import random
import numpy as np


def generateTestTrainindexes(data_length,testfrac):

    test_length=int(round(testfrac*data_length))
    train_length=data_length-test_length
    indexes=np.array((range(0,data_length)))

    random.shuffle(indexes)
    train_indexes=indexes[0:train_length]
    test_indexes = indexes[train_length:]

    return train_indexes,test_indexes

def generateTestTrainCrossover(data_length,testfrac,Nb_runs):
    return [generateTestTrainindexes(data_length,testfrac) for i in range(Nb_runs)]

train_indexes,test_indexes=generateTestTrainindexes(6,.3)
groups=generateTestTrainCrossover(6,.3,2)

print("Train indexes: {ind}\n".format(ind = train_indexes))
print("Test indexes: {ind}\n".format(ind = test_indexes))

print("Test -train groups: {ind}\n".format(ind = groups))