from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.form.get('berita')

    #preprocessing

    #n-gram

    #tf-idf

    #predict

    return data
    

if __name__ == '__main__':
  app.run(debug=True)     

