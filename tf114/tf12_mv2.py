import tensorflow as tf
tf.compat.v1.set_random_seed(66)

#1. 데이터

x_data = [[73, 51, 65],                       # (5, 3)
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]

y_data = [[152],[185],[180],[205],[142]]      # (5, 1)                      

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])         
                                                                     # 행열 연산은 앞의 열과 뒤의 행의 shape이 맞아야 함 w의 행은 x의 열의 갯수 와 동일한 shape이여야 함  
w = tf.compat.v1.Variable(tf.random.normal([3,1]), name='weight')    # y = x * w  ;  (5, 1) = (5, 3) * (? * ?)  => (3, 1)    
b = tf.compat.v1.Variable(tf.random.normal([1]), name='bias')        # bias는 덧셈이므로 shape변화 없음

#2. 모델
# hypothesis = x * w + b
hypothesis = tf.matmul(x, w) + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(10001):
    _, loss_val, w_val= sess.run([train, loss, w], feed_dict={x:x_data, y:y_data})
    print(epochs, '\t', loss_val, '\t', w_val)
    
   
#4. 예측
predict = tf.matmul(x, w_val) + b   # predict = model.predict

y_predict = sess.run(predict, feed_dict={x:x_data, y:y_data})
print("예측 : " , y_predict)

sess.close()

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_data, y_predict)
print('r2스코어 : ', r2)

mae = mean_absolute_error(y_data, y_predict)
print('mae : ', mae)

'''
예측 :  [[171.28122]
 [191.20404]
 [151.99522]
 [212.73956]
 [127.17381]]
r2스코어 :  0.4370552609459193
mae :  15.2111572265625
'''