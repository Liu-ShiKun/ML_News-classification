from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
import jieba
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle 
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
warnings.filterwarnings("ignore")


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    train_df = pd.DataFrame({'分类':['体育','健康','女人','娱乐','房地产','教育','文化','新闻','旅游','汽车','科技','财经']},columns = ['分类'])
    labelEncoder = LabelEncoder()
    y=labelEncoder.fit_transform(train_df['分类'])
    pickleFilePath = 'saved_variable/word2vec_model.pickle'
    with open(pickleFilePath, 'rb') as file:
        word2vec_model = pickle.load(file)

    pickleFilePath = 'saved_variable/logisticRegression_model.pickle'
    with open(pickleFilePath, 'rb') as file:
        logisticRegression_model = pickle.load(file)

    def get_contentVector(cutWords, word2vec_model):
        vector_list = [word2vec_model.wv[k] for k in cutWords if k in word2vec_model]
        contentVector = np.array(vector_list).mean(axis=0)
        return contentVector

    def get_featureMatrix(content_series):
        vector_list = []
        for content in content_series:
            vector = get_contentVector(jieba.cut(content, True), word2vec_model)
            vector_list.append(vector)
        featureMatrix = np.array(vector_list)
        return featureMatrix

    if request.method == 'POST':
        message = request.form['message']
        data = message

        new_df = pd.DataFrame({'分类':['不确定'],'内容':['等待输入']},columns = ['分类', '内容'])
        new_df['内容'][0] = data
        new_featureMatrix = get_featureMatrix(new_df['内容'])
        predict_label = logisticRegression_model.predict(new_featureMatrix)
        predict_return=labelEncoder.inverse_transform(predict_label)[0]
    return render_template('result.html',prediction = predict_return)


if __name__ == '__main__':
	app.run(debug=True)