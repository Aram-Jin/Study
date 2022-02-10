import tensorflow as tf
tf.compat.v1.set_random_seed(66)
                     # input_dim=1
변수 = tf.Variable(tf.random.normal([1]), name='weight')
print(변수)   # <tf.Variable 'Variable:0' shape=(1,) dtype=float32_ref>

#1. 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(변수)      
print("aaa : ", aaa)    # aaa :  [0.06524777]
sess.close()

#2. 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session=sess)      
print("bbb : ", bbb)    # bbb :  [0.06524777]

#3. 
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval()    
print("ccc : ", ccc)    # ccc :  [0.06524777]