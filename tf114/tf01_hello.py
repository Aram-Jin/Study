import tensorflow as tf                                   # tf.constant 상수 / tf.variable 변수 / tf.placeholder
print(tf.__version__)   #1.14.0                           # constant 상수 : 변하지 않는 숫자를 의미함 -> constant()함수를 통해서 정의함
                                                          # 통상적으로 상수는 대문자로 정의함 / 변수는 소문자 사용
# print('hello world')
                                          
hello = tf.constant('Hello World')         
# print(hello)           # Tensor("Const:0", shape=(), dtype=string)                     

# sess = tf.Session()                                        # 텐서플로우는 연산방식이 그래프 연산(노드연산) 
sess = tf.compat.v1.Session()
print(sess.run(hello))   # b'Hello World'                  # "sess run"을 통해 결과값을 출력해야함


# https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=complusblog&logNo=221237818389