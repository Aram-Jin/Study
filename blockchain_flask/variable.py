from flask import Flask, render_template
app = Flask(__name__)


@app.route('/hello/<user>')    # 변수값을 인자로 받는다
def hello_name(user):   # 변수값을 인자로 받는다
    return render_template('variable.html', name1=user, name2=2)  # html 템플릿에 변수값을 적용한다. 여러개의 변수를 전달할 수 있다
    #'variable.html'페이지에 변수넣어서 html을 return한다..


@app.route('/hello2/<user1>/<user2>')    # 변수값을 인자로 받는다
def hello_name2(user1, user2):   # 변수값을 인자로 받는다
    return render_template('variable.html', name1=user1, name2=user2)  # html 템플릿에 변수값을 적용한다. 여러개의 변수를 전달할 수 있다
    #'variable.html'페이지에 변수넣어서 html을 return한다..

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8080")