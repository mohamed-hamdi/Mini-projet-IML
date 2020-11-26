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


class TrainTestGenerator:

    """Iterator that counts upward forever."""

    def __init__(self, data_length,testfrac,Nb_runs,data=None):
        self.data_length = data_length
        self.testfrac = testfrac
        self.num = 0
        self.Nb_runs = Nb_runs
        self.data = data
        if data is not None:
                self.data_length = data.shape[0]
        print(self.data_length )

    def __iter__(self):
        return self

    def __next__(self):
        if self.num!=self.Nb_runs:
            self.num += 1
            if self.data is None:

                return generateTestTrainindexes(self.data_length, self.testfrac)
            else:
                train_indexes, test_indexes=generateTestTrainindexes(self.data_length, self.testfrac)
                return np.take(self.data, train_indexes, axis=0),np.take(self.data, test_indexes, axis=0)


        else:
            raise StopIteration



train_indexes,test_indexes=generateTestTrainindexes(6,.3)
groups=generateTestTrainCrossover(6,.3,2)


#print("Train indexes: {ind}\n".format(ind = train_indexes))
#print("Test indexes: {ind}\n".format(ind = test_indexes))

#print("Test -train groups: {ind}\n".format(ind = groups))

sample_data=np.array([[1.4, 2.2] ,[1.8 ,4.5],[1.2,6.2]])
gen=TrainTestGenerator(2,.3,2,data=sample_data)

for test,train in gen:
    print("Train indexes: {ind}\n".format(ind=train))
    print("Test indexes: {ind}\n".format(ind=test))

