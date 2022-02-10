# 실습
# 1. [4]
# 2. [5, 6]
# 3. [6, 7, 8] 

# 위 값들을 이용해서 predict해보기
# x_test라는 placeholder 생성!!

# y = wx + b
import tensorflow as tf
tf.set_random_seed(77)

#1. 데이터
x_train = tf.placeholder(tf.float32, shape=[None])  
y_train = tf.placeholder(tf.float32, shape=[None])  
x_test = tf.placeholder(tf.float32, shape=[None])  

w = tf.Variable(tf.random.normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random.normal([1]), dtype=tf.float32)

#2. 모델구성
hypothesis = x_train * w + b      # y = wx + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))   # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)       # optimizer='sgd'
# model.compile(loss='mse', optimizer='sgd')

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(2001):
    # sess.run(train)
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b], 
                                        feed_dict={x_train:[1,2,3], y_train:[1,2,3]})
    if step % 20 == 0:
        # print(step, sess.run(loss), sess.run(w), sess.run(b))
        print(step, loss_val, w_val, b_val)
    
#4. 예측
predict = x_test * w_val + b_val      # predict = model.predict

print("[4] 예측 : " , sess.run(predict, feed_dict={x_test:[4]}))
print("[5,6] 예측 : " , sess.run(predict, feed_dict={x_test:[5,6]}))
print("[6,7,8] 예측 : " , sess.run(predict, feed_dict={x_test:[6,7,8]}))

sess.close()


'''
[4] 예측 :  [3.9892442]
[5,6] 예측 :  [4.9830155 5.976787 ]
[6,7,8] 예측 :  [5.976787  6.970558  7.9643292]
'''



