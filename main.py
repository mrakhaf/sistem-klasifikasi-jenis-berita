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

    print()
    print("=====================================================================================")
    print('data')
    print(data)

    #preprocessing
    data = preprocessing(data)
    print()
    print("=====================================================================================")
    print('preprocessing')
    print(data)


    #n-gram
    data = n_gram(data)
    print()
    print("=====================================================================================")
    print('n-gram')
    print(data[1])



    #tf-idf
    data = count_tfidf([data[1]],bow_path,idf_path)
    print()
    print("=====================================================================================")
    print('tf-idf')
    print(data)


    #predict
    loaded_model = joblib.load(saved_model_path)
    proba = loaded_model.predict_proba(data)[0]    
    data = loaded_model.predict(data)[0]
    print()
    print("=====================================================================================")
    print('predict')
    print(data)
    print('Probality untuk label entertaiment : ')
    print("%.100f" % proba[0])
    print('Probality untuk label olahraga : ')
    print("%.100f" % proba[1])
    print()
    print()
    print()



    return json.dumps({
        'data': data
        })
    

if __name__ == '__main__':
  app.run(debug=True)     

