###########################
# Title - Linear regression using Tensorflow
############################

#Dependencies
import tensorflow as tf
import  numpy as np
import pylab

#Create input data and labels with some noise
x_data = np.random.rand(100).astype(np.float32)
noise = np.random.normal(scale=0.01,size=len(x_data))
y_data = x_data * 0.1 + 0.3 + noise

#Plot our data
pylab.figure("Datapoints")
pylab.plot(x_data,y_data,'.')
pylab.show()

#Weights and biases
W = tf.Variable(tf.random_uniform([1],0.0,1.0))
b = tf.Variable(tf.zeros([1]))

#Build Inference Graph
y = W * x_data + b
print(W)
print(b)

#Build Training Graph
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()


#create a session and launch the graph
sess = tf.Session()
sess.run(init)
y_initial = sess.run(y)
for i in range(101):
    sess.run(train)
    if(i%10==0):
        print(i,sess.run([W,b]))
print(sess.run([W,b]))

pylab.figure("Regression")
pylab.plot(x_data,y_data,'.',label='Target values')
pylab.plot(x_data,y_initial,'.',label='Initial values')
pylab.plot(x_data,sess.run(y))
pylab.legend()
pylab.ylim(0,1.0)
pylab.show()




