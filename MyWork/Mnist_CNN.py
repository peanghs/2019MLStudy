import tensorflow as tf
import matplotlib.pyplot as plt
import random
from datetime import datetime

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

#########
# 신경망 모델 구성
######
# 기존 모델에서는 입력 값을 28x28 하나의 차원으로 구성하였으나,
# CNN 모델을 사용하기 위해 2차원 평면과 특성치(흑백)의 형태를 갖는 구조로 만듭니다.
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# 각각의 변수와 레이어는 다음과 같은 형태로 구성됩니다.
# W1 [3 3 1 32] -> [3 3]: 커널 크기, 1: 입력값 X 의 특성수, 32: 필터 갯수
# L1 Conv shape=(?, 28, 28, 32)
#    Pool     ->(?, 14, 14, 32)
with tf.variable_scope('Layer1') as scope:
    W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
    # tf.nn.conv2d 를 이용해 한칸씩 움직이는 컨볼루션 레이어를 쉽게 만들 수 있습니다.
    # padding='SAME' 은 커널 슬라이딩시 최외곽에서 한칸 밖으로 더 움직이는 옵션 -> 크기를 똑같이 유지시키는 옵션
    # 원래라면 (커널 크기 -1) 만큼 차원이 감소
    # stride = [batch<-이미지 건너뛰기, width, height, depth<-색상]
    # http://imsjhong.blogspot.com/2017/07/tensorflow-stride-reshape.html
    # https://stackoverflow.com/questions/34642595/tensorflow-strides-argument
    L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    # Pooling 역시 tf.nn.max_pool 을 이용하여 쉽게 구성할 수 있습니다.
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    tf.summary.histogram('X', X)
    tf.summary.histogram('weight1', W1)
    tf.summary.histogram('layer1', L1)
    # L1 = tf.nn.dropout(L1, keep_prob)

# L2 Conv shape=(?, 14, 14, 64)
#    Pool     ->(?, 7, 7, 64)
# W2 의 [3, 3, 32, 64] 에서 32 는 L1 에서 출력된 W1 의 마지막 차원, 필터의 크기 입니다.
with tf.variable_scope('Layer2') as scope:
    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))

    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    tf.summary.histogram('weight2', W2)
    tf.summary.histogram('layer2', L2)
    # L2 = tf.nn.dropout(L2, keep_prob)

# FC 레이어: 입력값 7x7x64 = 3136 -> 출력값 256
# Full connect를 위해 직전의 Pool 사이즈인 (?, 7, 7, 64) 를 참고하여 차원을 줄여줍니다.
#    Reshape  ->(?, 256)
with tf.variable_scope('Layer3_FC') as scope:
    W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))

    L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
    L3 = tf.matmul(L3, W3)
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.dropout(L3, keep_prob)

    tf.summary.histogram('weight3', W3)
    tf.summary.histogram('layer3', L3)

# 최종 출력값 L3 에서의 출력 256개를 입력값으로 받아서 0~9 레이블인 10개의 출력값을 만듭니다.
with tf.variable_scope('Layer4_Output') as scope:
    W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
    b4 = tf.Variable(tf.random_normal([10]))
    model = tf.matmul(L3, W4) + b4

    tf.summary.histogram('weight4', W4)
    tf.summary.histogram('bias', b4)
    tf.summary.histogram('model', model)

with tf.variable_scope('cost') as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
    tf.summary.scalar('cost', cost)

optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
with tf.variable_scope('accuracy') as scope:
    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

merged_summary = tf.summary.merge_all()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#########
# 신경망 모델 학습
######
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = tf.summary.FileWriter("./logs/Mnist_CNN_log_%s/" % timestamp)
folder = "C:/Users/PHS/PycharmProjects/ML_Basic/MyWork/logs/Mnist_CNN_log_%s/" % timestamp
loots = "C:/Users/PHS/PycharmProjects/ML_Basic/MyWork/logs/"
writer.add_graph(sess.graph)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 이미지 데이터를 CNN 모델을 위한 자료형태인 [28 28 1] 의 형태로 재구성합니다.
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        s, _ = sess.run([merged_summary, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
        writer.add_summary(s, global_step=i)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

#########
# 결과 확인
######
print('최적화 완료!')
print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1), Y: mnist.test.labels,
                                            keep_prob: 1}))

#########
# 이미지 확인
######

for i in range(3):
    r = random.randint(0, mnist.test.num_examples - 1)
    print_xs = mnist.test.images[r:r + 1].reshape(-1, 28, 28, 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(tf.argmax(model, 1), feed_dict={X: print_xs, keep_prob: 1.0}))
    plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()

print('tensorboard --logdir=%s' % folder)
print('tensorboard --logdir=%s' % loots)
print('localhost:6006 를 크롬에 입력하세요')
