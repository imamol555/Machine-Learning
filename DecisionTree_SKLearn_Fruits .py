#===================================================
#Title  : Decision Tree Classifier for classifying a type of fruit
#Author : Amol Patil
#
#=======================================
#classes - Apple and Orange
#Features = [weight, texture]
#Texture- Smooth =1
#         Bumpy =0
#Labels - Apple =0
#         orange=1

from sklearn import tree

features = [[140,1],[130,1],[150,0],[170,0]]
labels = [0,0,1,1]

#selecting a type of classfier
#initially a classifier is nothing but a box of rules

clf = tree.DecisionTreeClassifier()
#Now the classifier will find a pattern to classify our data
#fit() method takes our train data and builds up a classifier
#it's a training algorithm which can be applied on classifier object clf
#creating a decision tree ie. setting up the classifier rules
clf = clf.fit(features,labels)

#predict for a random data
result = clf.predict([[152,0],[139,0]])
print(result)
for i in result:
    if (i==1):print("Orange")
    else:print("Apple")
