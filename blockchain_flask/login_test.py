from flask import Flask, jsonify, request, render_template
app = Flask(__name__, static_url_path='/static')

@app.route('/login')
def login():
    username = request.args.get('user_name')
    passwd = request.args.get('pw')
    email = request.args.get('email_address')
    
    if username == 'dave':
        return_data = {'auth' : 'sucess'}
    else:
        return_data = {'auth' : 'failed'}
    return jsonify(return_data)    
    
    
@app.route('/html_test')
def hello_html():
    return render_template('login.html') 

@app.route('/html_test1')
def hello_html1():
    return render_template('login_raw.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080')
     