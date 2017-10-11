from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

#load iris dataset
iris = datasets.load_iris()

#create an object of GaussianNB classifier
clf = GaussianNB()

#now train the model
my_model = clf.fit(iris.data, iris.target) #returns a trained model
predictions = my_model.predict(iris.data) # returns pedictions

#now we see the accuracy of model against
# the trained data itself
print("Number of mislabeled points out \
        of a total %d points : %d"  \
          % (iris.data.shape[0],\
             (iris.target != predictions).sum()))
