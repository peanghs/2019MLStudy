import tensorflow as tf
import random
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
B1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + B1)

W1_his = tf.summary.histogram('weight1_h', W1)
B1_his = tf.summary.histogram('bias1_h', B1)
L1_his = tf.summary.histogram('layer1_h', L1)

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
B2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2)+B2)

W2_his = tf.summary.histogram('weight2_h', W2)
B2_his = tf.summary.histogram('bias2_h', B2)
L2_his = tf.summary.histogram('layer2_h', L2)

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
B3 = tf.Variable(tf.random_normal([10]))
model = tf.matmul(L2, W3) + B3

W3_his = tf.summary.histogram('weight3_h', W3)
B3_his = tf.summary.histogram('bias3_h', B3)
L3_his = tf.summary.histogram('model', model)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
cost_sum = tf.summary.scalar('cost', cost)
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
accuracy_sum = tf.summary.scalar("accuracy", accuracy)

init = tf.global_variables_initializer()
sess = tf.Session()

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/xor_logs_r0_01")
writer.add_graph(sess.graph)

sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        summary, _ = sess.run([merged_summary, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
        writer.add_summary(summary, global_step=i)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')
print('정확도:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

for i in range(5):
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(model, 1), feed_dict={X: mnist.test.images[r:r + 1]}))
    plt.imshow(
        mnist.test.images[r:r + 1].reshape(28, 28),
        cmap='Greys',
        interpolation='nearest')
    plt.show()

print('localhost:6006 를 크롬에 입력하세요')
