import random
import numpy as np




#def generateTestTrainCrossover(data_length,testfrac,Nb_runs):
#    return [generateTestTrainindexes(data_length,testfrac) for i in range(Nb_runs)]


class TrainTestGenerator:

    """Iterator that counts upward forever."""

    def __init__(self, n_splits,test_size,train_size,data=None,data_length=None):
        self.data_length = data_length
        self.test_size = test_size
        self.train_size = train_size
        self.num = 0
        self.n_splits = n_splits
        self.data = data
        if data is not None:
                self.data_length = data.shape[0]

    def generateTestTrainindexes(self,data_length, test_size=None, train_size=None):

        if test_size is not None:
            if isinstance(test_size, float):
                test_length = int(round(test_size * data_length))
            else:
                test_length = test_size
            train_length = data_length - test_length

        if train_size is not None:
            if isinstance(train_size, float):
                train_length = int(round(train_size * data_length))
            else:
                train_length = train_size

        indexes = np.array((range(0, data_length)))

        random.shuffle(indexes)
        train_indexes = indexes[0:train_length]
        test_indexes = indexes[train_length:]

        return train_indexes, test_indexes
    def __iter__(self):
        return self

    def __next__(self):
        if self.num!=self.n_splits:
            self.num += 1
            print(self.test_size)
            return self.generateTestTrainindexes(self.data_length, test_size=self.test_size, train_size=self.train_size)
            #if self.data is None:

            #    return self.generateTestTrainindexes(self.data_length, test_size=self.test_size,train_size=self.train_size)
            #else:
            #    train_indexes, test_indexes=self.generateTestTrainindexes(self.data_length, test_size=self.test_size,train_size=self.train_size)
            #    return np.take(self.data, train_indexes, axis=0),np.take(self.data, test_indexes, axis=0)


        else:
            raise StopIteration



class CustomShuffle():
    def __init__(self, n_splits,test_size=None,train_size=None,data=None,data_length=None):
        self.data_length = data_length
        self.test_size = test_size
        self.train_size = train_size
        self.num = 0
        self.n_splits = n_splits
        self.data = data

    def split(self,data):
        return TrainTestGenerator(n_splits=self.n_splits, test_size=self.test_size, train_size=self.train_size, data=data)



#print("Train indexes: {ind}\n".format(ind = train_indexes))
#print("Test indexes: {ind}\n".format(ind = test_indexes))

#print("Test -train groups: {ind}\n".format(ind = groups))

sample_data=np.array([[1.4, 2.2] ,[1.8 ,4.5],[1.2,6.2]])
sample_data=np.array([1.4, 2.2 ,1.8 ,4.5,1.2,6.2])

myshuffler=CustomShuffle(2,test_size=0.8)

for train,test in myshuffler.split(sample_data):
    print("#################################################################")
    print("Train indexes: {ind}\n".format(ind=train))
    print("Test indexes: {ind}\n".format(ind=test))
    print("#################################################################")

