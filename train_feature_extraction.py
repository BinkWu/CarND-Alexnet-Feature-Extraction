import pickle as pk
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet
import numpy as np

# TODO: Load traffic signs data.
with open('./train.p','rb') as f:
    t_data = pk.load(f)

# TODO: Split data into training and validation sets.
n_classes = np.size(np.unique(t_data['labels']))

train_x,test_x,train_y,test_y = train_test_split(t_data['features'],t_data['labels'],test_size=0.33)
# train_x = tf.image.resize_images(train_x,(227,227))
# test_x = tf.image.resize_images(test_x,(227,227))
# TODO: Define placeholders and resize operation.
batch_size = 64
graph = tf.Graph()
with graph.as_default():
    batch_x = tf.placeholder(shape=(None, 32, 32,3), dtype=tf.float32)
    # batch_x = tf.placeholder(shape=(None,227,227,3),dtype=tf.float32)
    batch_y = tf.placeholder(tf.int64, None)
    resized = tf.image.resize_images(batch_x,(227,227))

    y = tf.one_hot(batch_y,n_classes)
    # TODO: pass placeholder as first argument to `AlexNet`.
    fc7 = AlexNet(resized, feature_extract=True)
    # NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
    # past this point, keeping the weights before and up to `fc7` frozen.
    # This also makes training faster, less work to do!
    fc7 = tf.stop_gradient(fc7)

    # TODO: Add the final layer for traffic sign classification.
    fc8_w = tf.Variable(tf.truncated_normal(shape=(4096,n_classes),stddev=1e-2))
    fc8_b = tf.Variable(tf.zeros((n_classes)))
    fc8 = tf.nn.xw_plus_b(fc7,fc8_w,fc8_b)
    # TODO: Define loss, training, accuracy operations.
    # HINT: Look back at your traffic signs project solution, you may
    # be able to reuse some the code.
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=fc8))
    global_step = tf.Variable(0)
    lr = 0.001
    dr = 0.96
    learning_rate = tf.train.exponential_decay(
            learning_rate=lr,
            global_step=global_step*batch_size,
            decay_steps=20,
            decay_rate=dr,
            staircase=True
    )
    operator = tf.train.AdamOptimizer().minimize(loss)

    predicts = tf.nn.softmax(fc8)
    correct_prediction = tf.equal(tf.argmax(predicts,1), tf.argmax(y,1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TODO: Train and evaluate the feature extraction model.
def evaluate(i_X,i_Y,sess):
    total_a = 0
    num_sample = len(i_X)
    for k in range(0,len(i_X),batch_size):
        end = k + batch_size
        b_x, b_y = i_X[k:end], i_Y[k:end]
        l,acc = sess.run([loss,accuracy_operation], {batch_x: b_x, batch_y: b_y})
        total_a += (acc * b_x.shape[0])
    return total_a/num_sample

epoch = 10
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    for i in range(epoch):
        for k in range(0,train_x.shape[0],batch_size):
            end = k+batch_size
            b_x,b_y = train_x[k:end],train_y[k:end]
            l,o,accuracy = sess.run([loss,operator,accuracy_operation],{batch_x:b_x,batch_y:b_y})
        print('epoch:',i)
        print('the accuracy of training set is' , evaluate(train_x,train_y,sess))
        print('the accuracy of testing set is' , evaluate(test_x,test_y,sess))

