import tensorflow as tf
sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype=tf.float32)

init = tf.compat.v1.global_variables_initializer()   # -> 초기화 함수 ; 변수 초기화
sess.run(init)                                       # 변수 초기화 수행

print("잘나오니? ", sess.run(x))   # [2.]