import tensorflow as tf

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

A = tf.sigmoid(tf.matmul(X, W) + b)  # matmul 행렬곱 한 시점에서 차원은 1차원 6개 항 이 된듯
cost = -tf.reduce_mean(Y * tf.log(A) + (1 - Y) * tf.log(1 - A))  # 1차원 곱인데 reduce mean 은 왜 했지..?
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)  # learning_rate 는 스텝 사이즈

output = tf.cast(A > 0.5, dtype=tf.float32)  # int 32 넣어도 무방한가? 어차피 결과값이 1.0 / 0.0 형태라면 1 / 0 이어도 상관없는거 아닌가
accuracy = tf.reduce_mean(tf.cast(tf.equal(output, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})  # 의미 파악 필요
        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([A, output, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis : ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
