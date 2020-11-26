import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler




class PreProcessor:
    pca=None
    scaler=None

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

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])


features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = df.loc[:, features].values


# Separating out the target
y = df.loc[:,['target']].values



#preProc = PreProcessor(NbComponents=2)
preProc = PreProcessor(retained_variance=0.99999)

principalComponents = preProc.fit_transform(x)
print(principalComponents)



# principalDf = pd.DataFrame(data = principalComponents
#              , columns = ['principal component 1', 'principal component 2'])
#
# finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
# print(finalDf)
#
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
# targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# colors = ['r', 'g', 'b']
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['target'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                , finalDf.loc[indicesToKeep, 'principal component 2']
#                , c = color
#                , s = 50)
# ax.legend(targets)
# ax.grid()
#
# plt.show()