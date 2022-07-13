from flask import Flask, request, render_template
import pandas as pd
from utils.preprocessing import preprocessing
from utils.n_gram import n_gram
from utils.tf_idf import count_tfidf
import joblib

bow_path = '' #lokasi file pickle bag_of_word
idf_path = '' #lokasi file pickle idf
saved_model_path = ''

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
    data = count_tfidf([data],bow_path,idf_path)

    #predict
    loaded_model = joblib.load(saved_model_path)
    data = loaded_model.predict(data)[0]

    return str(data)
    

if __name__ == '__main__':
  app.run(debug=True)     

