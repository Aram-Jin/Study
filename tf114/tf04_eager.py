import tensorflow as tf
print(tf.__version__)   # 1.14.0
print(tf.executing_eagerly())   # False

# 즉시실행모드
tf.compat.v1.disable_eager_execution()  # 꺼!!

print(tf.executing_eagerly())    # False

hello = tf.constant("Hello World")

sess = tf.compat.v1.Session()
print(sess.run(hello))    # b'Hello World'

