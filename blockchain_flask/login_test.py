from flask import Flask, jsonify, request, render_template  # request : url에 parameter를 넣어서 요청하려면 request 모듈이 필요하다..
app = Flask(__name__, static_url_path='/static')  # static_url_path='/static' : 모든 관련 파일의 가져올 위치


@app.route('/login')            # api를 생성 : 요쳥주소, 요청 파라미터,.. => rest api
def login():
    username = request.args.get('user_name')   
    # get방식으로 요청을 한 경우 url에 파라미터값을 넣어서 요청하는데 
    # 그 파라미터 값을 가져오기 위해 request.args.get('파라미터') 메소드를 사용한다.
    passwd = request.args.get('pw')
    email = request.args.get('email_address')   # param 값을 받을 수 있다
    print (username, passwd, email)  # url에 param으로 요청한 값들이 출력된다.
    
    if username == 'dave':
        return_data = {'auth': 'success'}
    else:
        return_data = {'auth': 'failed'}
    return jsonify(return_data)  # front_end에서 json data를 받을 수 있다.

@app.route('/html_test')     # 요청할 수 있는 페이지
def hello_html():
    # html file은 templates 폴더에 위치해야 함
    return render_template('login.html')   # templates 폴더 안에서 login.html 파일을 찾는다.

@app.route('/html_test1')  # 경로를 요청하면
def hello_html1():   # 함수를 호출
    # html file은 templates 폴더에 위치해야 함
    return render_template('login_raw.html')   # templates 폴더 안에서 login_raw.html 파일을 찾는다.

if __name__ == '__main__':       # 웹서버 구동
    app.run(host="0.0.0.0", port="8080")
