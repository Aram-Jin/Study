import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.set_random_seed(66)

x_train_data = [1,2,3]
y_train_data = [1,2,3]
x_test_data = [4,5,6]
y_test_data = [4,5,6]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)
x_test = tf.placeholder(tf.float32, shape=[None])  
y_test = tf.placeholder(tf.float32, shape=[None])  

w = tf.compat.v1.Variable(tf.random_normal([1]), name='weight')

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))

lr = 0.1
gradient = tf.reduce_mean((w * x - y) * x)
descent = w - lr * gradient
update = w.assign(descent)      # w = w - lr * gradient

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# w_history = []
# loss_history = []

for step in range(21):
    
    # sess.run(update, feed_dict={x:x_train, y:y_train})
    # print(step, '\t', sess.run(loss, feed_dict={x:x_train, y:y_train}), '\t', sess.run(w))
    
    _, loss_val, w_val = sess.run([update, loss, w], feed_dict={x:x_train_data, y:y_train_data})
    print(step, '\t', loss_val, '\t', w_val)
    
    # w_history.append(w_val)
    # loss_history.append(loss_val)
   
#4. 예측
predict = x_test * w_val      # predict = model.predict

y_predict = sess.run(predict, feed_dict={x_test:x_test_data})
print("[4,5,6] 예측 : " , y_predict)

sess.close()
   
from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_test_data, y_predict)
print('r2스코어 : ', r2)

mae = mean_absolute_error(y_test_data, y_predict)
print('mae : ', mae)

'''
[4,5,6] 예측 :  [3.99999   4.9999876 5.9999847]
r2스코어 :  0.9999999997565965
mae :  1.2556711832682291e-05
'''