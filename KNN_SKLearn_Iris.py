#########################################
#Title - Implementation of K-Nearest Neighbour in Python
#Author - Amol Patil
#Dataset - Iris Flower Dataset
############################################
from sklearn.datasets import load_iris
iris = load_iris()
from scipy.spatial import distance

#Define a function to calulate euclidean distance between feature values
def euc(a,b):
    return distance.euclidean(a,b)
#Class for KNN Classifier
class MyKNN():
    def fit(self,X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self,X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self,row):
        best_dist = euc(row,self.X_train[0])
        best_index = 0
        for i in range(1,len(self.X_train)):
            dist = euc(row,self.X_train[i])
            if(dist < best_dist):
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

#defining variables    
X = iris.data
y = iris.target

#split the data into two halves- train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

#create a model
clf = MyKNN()

#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier()   # accuracy 0.9466

clf.fit(X_train,y_train)

#predict for test data
predictions = clf.predict(X_test)

#calculate the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))

