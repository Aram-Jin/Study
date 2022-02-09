# # 실습
# lr수정해서 epochs를 100번 이하로 줄여라!!
# step = 100 이하, w = 1.99, b = 0.99

# y = wx + b
import tensorflow as tf
tf.set_random_seed(77)

# 1. 데이터
x_train_data = [1, 2, 3]
y_train_data = [3, 5, 7]

x_train = tf.placeholder(tf.float32, shape=[None])  
y_train = tf.placeholder(tf.float32, shape=[None])  
x_test = tf.placeholder(tf.float32, shape=[None])  

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

#2. 모델구성
hypothesis = x_train * w + b      # y = wx + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))   # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.15)
train = optimizer.minimize(loss)       # optimizer='sgd'
# model.compile(loss='mse', optimizer='sgd')

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(100):
    # sess.run(train)
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b], 
                                        feed_dict={x_train:x_train_data, y_train:y_train_data})
    if step % 20 == 0:
        # print(step, sess.run(loss), sess.run(w), sess.run(b))
        print(step, loss_val, w_val, b_val)
    
#4. 예측

predict = x_test * w_val + b_val      # predict = model.predict

print("[6,7,8] 예측 : " , sess.run(predict, feed_dict={x_test:[6,7,8]}))

sess.close()

'''
0 6.817572 [2.1196995] [2.5242271]
20 0.054019306 [1.7399766] [1.5915923]
40 0.012432656 [1.8751675] [1.2837739]
60 0.0028614288 [1.9401125] [1.1361387]
80 0.0006585773 [1.9712692] [1.0653118]
[6,7,8] 예측 :  [12.946711 14.932412 16.918114]
'''




