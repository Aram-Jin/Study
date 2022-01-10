from flask import Flask, render_template
app = Flask(__name__)

@app.route('/hello_loop')
def hello_name():
    value_list = ['list1', 'list2', 'list3']
    # loop.html을 호출하면서, 값으로 list를 넘긴다..
    return render_template('loop.html', values=value_list)   # loop.html 은 templetes폴더에서 찾는다.

@app.route('/hello_loop1/<list1>')
def hello_name1(list1):
    value_list = eval(list1)
    # loop.html을 호출하면서, 값으로 list를 넘긴다..
    return render_template('loop.html', values=value_list)   # loop.html 은 templetes폴더에서 찾는다.

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8081")