import tensorflow as tf

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3  # 클래스 수

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))  # 합 하고 평균 내기
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if i % 200 == 0:
            print(i, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    print('---학습 끝---')

    a = sess.run(hypothesis, feed_dict={X: [[1, 2, 3, 4]]})
    print(a, '\n', sess.run(tf.argmax(a, 1)))
    print('------A 추정-------')

    b = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 8]]})
    print(b, '\n', sess.run(tf.argmax(b, 1)))
    print('------B 추정-------')

    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, '\n', sess.run(tf.argmax(c, 1)))
    print('------C 추정-------')

    all = sess.run(hypothesis, feed_dict={
                   X: [[1, 2, 3, 4], [1, 11, 7, 8], [1, 1, 0, 1]]})
    print(all, '\n', sess.run(tf.argmax(all, 1)))
    print('------다항 추정-------')
