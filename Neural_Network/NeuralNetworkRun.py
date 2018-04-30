import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
from sklearn.model_selection import train_test_split


# Load training and testing data
training_data = pd.read_pickle('/data/WorkData/firmEmbeddings/final_train_data.pkl');
testing_data = pd.read_pickle('/data/WorkData/firmEmbeddings/final_test_data.pkl');

# Target train and test
y_training = pd.read_pickle('/data/WorkData/firmEmbeddings/y_train.pkl');
y_testing = pd.read_pickle('/data/WorkData/firmEmbeddings/y_test.pkl');

#Convert the input data to array
X_train = training_data.values
X_test = testing_data.values

#Convert the label data to array
y_train = y_training.values
y_test = y_testing.values

# Dimensions of dataset
num_of_inputVectors = X_train.shape[0]
num_of_features = X_train.shape[1]

print(num_of_inputVectors)
print(num_of_features)


#Create placeholder for input and output
X_input = tf.placeholder(dtype=tf.float32,shape=[None,395])
y_output = tf.placeholder(dtype=tf.float32,shape=[None,1])

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()


#Model architecture parameters
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_target = 1

# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([num_of_features, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

#Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

#Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))

# Layer 4: Variables for hidden weights and biases
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))


# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))

# Output weights
W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))
bias_out = tf.Variable(bias_initializer([1]))

# Hidden layer
#wordVector_to_columns = tf.cast(wordVector_to_columns,tf.float32)
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X_input, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

# Output layer (transpose!)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, y_output))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# Session
net = tf.InteractiveSession()

#Init
net.run(tf.global_variables_initializer())

# Fit neural net
batch_size = 256
mse_train = []
mse_test = []

X_test_part = X_test[0:5000]
y_test_part = y_test[0:5000]


np.savetxt('actual.txt', y_test_part, fmt='%f')

prediction = np.zeros(5000)

#plt.ion()
#fig = plt.figure()
# Run
epochs = 10
for e in range(epochs):

    # Shuffle training data
    #shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    #X_train = X_train[shuffle_indices]
    #y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        
        # Run optimizer with batch
        net.run(opt, feed_dict={X_input: batch_x, y_output: batch_y})

        # Show progress
        if np.mod(i, 50) == 0:
            # MSE train and test
            #mse_train.append(net.run(mse, feed_dict={X_input: X_train, y_output: y_train}))
            mse_test.append(net.run(mse, feed_dict={X_input: X_test_part, y_output: y_test_part})) 
            
            #print(e)
            #print(i)
            #print('MSE Train: ', mse_train[-1])
            print('MSE Test: ', mse_test[-1])
            
            # Prediction
            pred = net.run(out, feed_dict={X_input: X_test_part})
            prediction = np.asarray(pred)
            np.savetxt('prediction.txt', prediction, fmt='%f')
            print("prediction is")
            print(pred)
            #ax1 = fig.add_subplot(111)
            #line1, = ax1.plot(y_test_part)
            #line2, = ax1.plot(pred)
            #plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            #plt.show()


            
