import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터
         # 첫번 두번 세번 네번 다섯번
x1_data = [73., 93., 89., 96., 73.]       # 국어                  .을 찍은 이유는 float형태로 나타내주기위해서
x2_data = [80., 88., 91., 98., 66.]       # 영어
x3_data = [75., 93., 90., 100., 70.]      # 수학
y_data = [152., 185., 180., 196., 142.]   # 환산점수

# x는 (5,3), y는 (5,1) 또는 (5,)
# y = x1 * w1 + x2 * w2 + x3 * w3

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w1 = tf.compat.v1.Variable(tf.random_normal([1]), name = 'weight1')
w2 = tf.compat.v1.Variable(tf.random_normal([1]), name = 'weight2')
w3 = tf.compat.v1.Variable(tf.random_normal([1]), name = 'weight3')
b = tf.compat.v1.Variable(tf.random_normal([1]), name = 'bias')

# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())
# print(sess.run([w1, w2, w3]))
# [array([-0.0209489], dtype=float32), array([0.4090447], dtype=float32), array([-0.9833048], dtype=float32)]

#2. 모델
hypothesis = x1*w1 + x2*w2 + x3*w3 + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(10001):
    _, loss_val, w_val1, w_val2, w_val3 = sess.run([train, loss, w1, w2, w3], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
    print(epochs, '\t', loss_val, '\t', w_val1, '\t', w_val2, '\t', w_val3)
    
   
#4. 예측
predict =  x1*w_val1 + x2*w_val2 + x3*w_val3 + b   # predict = model.predict

y_predict = sess.run(predict, feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
print("예측 : " , y_predict)

sess.close()

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_data, y_predict)
print('r2스코어 : ', r2)

mae = mean_absolute_error(y_data, y_predict)
print('mae : ', mae)

'''
예측 :  [151.9266  184.33585 180.84567 196.30974 141.42262]
r2스코어 :  0.9992438441759043
mae :  0.4940673828125
'''