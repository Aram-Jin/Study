import tensorflow as tf
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

# 실습 
# 덧셈 node3
# 뺄셈 node4
# 곱셈 node5
# 나눗셈 node6

# node3 = node1 + node2
node3 = tf.add(node1, node2)

# node4 = node1 - node2
node4 = tf.subtract(node1, node2)

# node5 = node1 * node2
node5 = tf.multiply(node1, node2)

# node6 = node1 / node2
node6 = tf.divide(node1, node2)

# sess = tf.Session()
sess = tf.compat.v1.Session()
print('node1, node2 :', sess.run([node1, node2]))  
print('node3 : ', sess.run(node3))   
print('node4 : ', sess.run(node4))   
print('node5 : ', sess.run(node5))   
print('node6 : ', sess.run(node6))   

# node1, node2 : [2.0, 3.0]
# node3 :  5.0
# node4 :  -1.0
# node5 :  6.0
# node6 :  0.6666667