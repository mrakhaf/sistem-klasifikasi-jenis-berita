import math
import mpu
import pandas as pd

def creating_bag_of_word():
    return mpu.io.read('bag_of_word.pickle')

def count_inv_doc_freq():
    return mpu.io.read('idf.pickle')

def count_log_term_freq(bow, doc_l):
    log_term_frequency = {}
    for b in bow:
        log_term_frequency[b] = {}
        for index,d in enumerate(doc_l):
            tfd = d.count(b)
            if tfd == 0:
                log_term_frequency[b][index] = 0
            else :
                log_term_frequency[b][index] = 1 + math.log10(tfd)
    return log_term_frequency

def count_tfidf(document_l):
    '''
    document_l : list dari dokumen yang sudah menjadi list dari ngram

    return : list dari hasil tfidf per dokumen

    '''
    doc_l = document_l
    bag_of_word = creating_bag_of_word()
    idf = count_inv_doc_freq() 
    tf = count_log_term_freq(bag_of_word,doc_l) 
    tfidf = tf

    for k in tf.keys():
        for i in tfidf[k].keys():
            tfidf[k][i] = tfidf[k][i] * idf[k]
    # tfidf = count_tfidf(df.bigram.tolist())
    df_tfidf = pd.DataFrame(tfidf)
    list_tfidf = df_tfidf.to_numpy().tolist()
    return list_tfidf