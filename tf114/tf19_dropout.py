import tensorflow as tf

# layers = tf.nn.dropout(Hidden_layer1, keep_prob=0.7)  # 남기고 싶은 퍼센트를 적어줌********
# 평가할 때는 dropout을 적용 안함 (evaluate단계에서는 빼줌)

#2. 모델구성
x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 1])
w1 = tf.compat.v1.Variable(tf.random.uniform([10,16]), name='weight1')    # y = x * w  
b1 = tf.compat.v1.Variable(tf.random.uniform([16]), name='bias1')   

Hidden_layer1 = tf.matmul(x, w1)+b1
layers = tf.nn.dropout(Hidden_layer1, keep_prob=0.7)  # 남기고 싶은 퍼센트를 적어줌********

w2 = tf.compat.v1.Variable(tf.random.uniform([16,12]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random.uniform([12]), name='bias2')

Hidden_layer2 = tf.nn.selu(tf.matmul(Hidden_layer1, w2)+b2)

w3 = tf.compat.v1.Variable(tf.random.normal([12,8]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random.normal([8]), name='bias3')

Hidden_layer3 = tf.matmul(Hidden_layer2, w3)+b3

w4 = tf.compat.v1.Variable(tf.random.normal([8,6]), name='weight4')
b4 = tf.compat.v1.Variable(tf.random.normal([6]), name='bias4')

Hidden_layer4 = tf.matmul(Hidden_layer3, w4)+b4

w5 = tf.compat.v1.Variable(tf.random.normal([6,1]), name='weight5')
b5 = tf.compat.v1.Variable(tf.random.normal([1]), name='bias5')

hypothesis = tf.matmul(Hidden_layer4, w5)+b5