import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.set_random_seed(66)

x_train = [1,2,3]
y_train = [1,2,3]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable(tf.random.normal([1]), name='weight')

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))

lr = 0.21
gradient = tf.reduce_mean((w * x - y) * x)
descent = w - lr * gradient
update = w.assign(descent)      # w = w - lr * gradient

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

w_history = []
loss_history = []

for step in range(21):
    
    # sess.run(update, feed_dict={x:x_train, y:y_train})
    # print(step, '\t', sess.run(loss, feed_dict={x:x_train, y:y_train}), '\t', sess.run(w))
    
    _, loss_val, w_val = sess.run([update, loss, w], feed_dict={x:x_train, y:y_train})
    print(step, '\t', loss_val, '\t', w_val)
    
    w_history.append(w_val)
    loss_history.append(loss_val)
    
sess.close()

print("======================================== w history =========================================")
print(w_history)
print("======================================== loss history =========================================")
print(loss_history)

plt.plot(w_history, loss_history)
plt.xlabel('weight')
plt.ylabel('loss')
plt.show()