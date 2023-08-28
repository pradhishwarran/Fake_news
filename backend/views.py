from django.shortcuts import render
import joblib 
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = joblib.load('./models/vectorizer.joblib')
lrmodel = joblib.load('./models/lrmodel.joblib')
dtmodel = joblib.load('./models/dtmodel.joblib')
gbmodel = joblib.load('./models/gbmodel.joblib')
rfmodel = joblib.load('./models/rfmodel.joblib')


def start(request):
    return render(request, "index.html")

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def output_lable(n):
    if n == 0:
        return ("Fake News")
    else:
        return ("Not A Fake News")

def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorizer.transform(new_x_test)
    pred_LR = lrmodel.predict(new_xv_test)
    pred_DT = dtmodel.predict(new_xv_test)
    pred_GB = gbmodel.predict(new_xv_test)
    pred_RF = rfmodel.predict(new_xv_test)

    print(pred_LR)
    print(pred_DT)
    print(pred_GB)
    print(pred_RF)

    global lrpred, dtpred, gbpred, rfpred 
    lrpred = output_lable(pred_LR.astype(int))
    dtpred = output_lable(pred_DT.astype(int))
    gbpred = output_lable(pred_GB.astype(int))
    rfpred = output_lable(pred_RF.astype(int))
    print(lrpred,dtpred,gbpred,rfpred)

def show(request):
    news = request.GET['news']
    manual_testing(news)
    return render(request, "output.html", {"lr": lrpred, "dt": dtpred, "gb": gbpred, "rf": rfpred})

