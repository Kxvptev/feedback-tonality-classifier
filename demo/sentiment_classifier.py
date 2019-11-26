# -*- coding: utf-8 -*-
from sklearn.externals import joblib


class SentimentClassifier(object):
    def __init__(self):
        self.model = joblib.load("model.pkl")
        self.vectorizer = joblib.load("vectorizer.pkl")
        self.classes_dict = {0: u"отзыв негативный", 1: u"отзыв позитивный", -1: u"ошибка предсказания"}

    def predict_text(self, text):
        try:
            vectorized = self.vectorizer.transform([text])
            return self.model.predict(vectorized)[0]
        except:
            print (u"ошибка предсказания")
            return -1

    # def predict_list(self, list_of_texts):
    #     try:
    #         vectorized = self.vectorizer.transform(list_of_texts)
    #         return self.model.predict(vectorized), \
    #                self.model.predict_proba(vectorized)
    #     except:
    #         print ('prediction error')
    #         return None

    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        return self.classes_dict[prediction]
