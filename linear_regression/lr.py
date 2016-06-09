import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

#Paramètres
#fera Y en fonction de X
# 0 = time; 1 = cwnd; 2 = rtt; 3 = dupack; 4 = retransmission; 5 = drops
axe_X = 1
axe_Y = 5
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data
fich = open('cwnd_reno.dat','r')
val = fich.read()
tmp = val.split('\r\n')
new_liste = list()
train_x = list()
train_y = list()
i = 1
while i < len(tmp)-1:
    tmp1 = tmp[i].split(';')
    train_x.append(float(tmp1[axe_X]))
    train_y.append(float(tmp1[axe_Y]))
    #mon_tuple = [float(tmp1[0]), float(tmp1[1])]
    #time & cwnd
    #liste_vide.append(mon_tuple)
    i += 150

fich.close()
train_X = numpy.asarray(train_x)
train_Y = numpy.asarray(train_y)

n_samples = train_X.shape[0]

# Test data
fich = open('cwnd_reno.dat','r')
val = fich.read()
tmp = val.split('\r\n')
liste_vide = list()
test_x = list()
test_y = list()
i = 1
while i < len(tmp)-1:
    tmp1 = tmp[i].split(';')
    test_x.append(tmp1[axe_X])
    test_y.append(tmp1[axe_Y])
    #mon_tuple = [float(tmp1[0]), float(tmp1[1])]
    #time & cwnd
    #liste_vide.append(mon_tuple)
    i += 10    fich.close()

test_X = numpy.asarray(test_x)
test_Y = numpy.asarray(test_y)
# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.mul(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b)

    print "Optimization Finished!"
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'

    #Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)

    print "Testing... (Mean square loss Comparison)"
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print "Testing cost=", testing_cost
    print "Absolute mean square loss difference:", abs(
        training_cost - testing_cost)

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
