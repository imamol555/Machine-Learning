#===================================================
#Title  : Decision Tree Classifier for classifying
#           species of Iris Flower
#Author : Amol Patil
#
#=======================================
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
'''
print(iris.feature_names)
print(iris.target_names)
print(type(iris.target))'''
#for i in range(len(iris.target)):
#  print("Exmaple %d, Label %s, Features %s"%(i,iris.target[i],iris.data[i]))

#print(type(iris.data)) --> np.ndarray object

"""We will take some instances from trainig data
as our test data"""
#take one instance each from  Setosa , Versicolor, Verginica
test_idx = [0,50,100]
#remove instances of test_idx from train data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data,test_idx,axis=0)

test_data = iris.data[test_idx]
test_target = iris.target[test_idx]

##########Training our Model############
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

#########Test our model##########
print(test_target)
print(clf.predict(test_data))



##########create a plot###########
import pydotplus
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("newiris.pdf")
import  os
os.startfile("newiris.pdf")
