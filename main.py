from flask import Flask, request, render_template
from utils.preprocessing import preprocessing
from utils.n_gram import n_gram
from utils.tf_idf import count_tfidf
import joblib
import json

bow_path = 'notebook/bag_of_word.pickle' #lokasi file pickle bag_of_word
idf_path = 'notebook/idf.pickle' #lokasi file pickle idf
saved_model_path = 'notebook/naivebayes_model.sav'

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
    data = count_tfidf([data[1]],bow_path,idf_path)

    #predict
    loaded_model = joblib.load(saved_model_path)
    data = loaded_model.predict(data)[0]

    return json.dumps({
        'data': data
        })
    

if __name__ == '__main__':
  app.run(debug=True)     

