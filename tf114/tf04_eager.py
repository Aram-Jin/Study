import tensorflow as tf
print(tf.__version__)   # 1.14.0
print(tf.executing_eagerly())   # False

# 즉시실행모드*****    => 텐서플로우2부터는 디폴트로 되어있기때문에 eager(즉시실행모드)를 사용하지 않음
tf.compat.v1.disable_eager_execution()  # 꺼!!

print(tf.executing_eagerly())    # False

hello = tf.constant("Hello World")

sess = tf.compat.v1.Session()
print(sess.run(hello))    # b'Hello World'

