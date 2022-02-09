import tensorflow as tf
import matplotlib.pyplot as plt

x = [1,2,3]
y = [1,2,3]
w = tf.placeholder(tf.float32)

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))

w_history = []
loss_history = []

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50):
        curr_w = i
        curr_loss = sess.run(loss, feed_dict={w:curr_w})
        
        w_history.append(curr_w)
        loss_history.append(curr_loss)
        
print("============================ w history ================================")
print(w_history)
print("============================ loss history ================================")
print(loss_history)

from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"   # 나눔고딕이 일반적으로 쓰는 글씨체
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

plt.plot(w_history, loss_history)
plt.xlabel('웨이트')
plt.ylabel('로스')
plt.title('선생님만세')
plt.show()

        