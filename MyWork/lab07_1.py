import tensorflow as tf
import random
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([10]))

L1 = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=L1, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

is_correct = tf.equal(tf.arg_max(L1, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 10
batch_size = 200

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('에폭:', '%d' % (epoch + 1), '비용 : ', '{:4f}'.format(avg_cost))

    print('----학습 종료----')

    # Test the model using test sets
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    # Get one and predict
    for i in range(5):
        r = random.randint(0, mnist.test.num_examples - 1)
        print("정답 : ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
        print("예측 : ", sess.run(tf.argmax(L1, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

        plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
        plt.show()
