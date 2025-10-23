#####
## index.py
#####

from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def index():
    name = "This is  the route page"
    return name

@app.route('/hello-world')
def hello_world():
    name = "hogehoge"
    
    num = 100
    print(num)
    num = num+  100
    print(num)
    num = num+  100
    print(num)
    num = num+  100
    print(num)
    return render_template('hello-world.html', title='Hello World!!', name=name)

if __name__ == "__main__":
    app.run()