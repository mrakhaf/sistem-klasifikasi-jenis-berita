from flask import Flask, request, render_template
import pandas as pd
from utils.preprocessing import preprocessing
from utils.n_gram import n_gram

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.form.get('berita')

    #preprocessing
    data = preprocessing(data)

    #n-gram
    data = n_gram(data)

    #tf-idf

    #predict

    return str(data)
    

if __name__ == '__main__':
  app.run(debug=True)     

