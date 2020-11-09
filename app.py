from flask import Flask,render_template,request,url_for
import pandas as pd
from pycaret.regression import *
import numpy as np
import pickle

app=Flask(__name__)
model = load_model('Housing')
cols=['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above','sqft_basement','yr_built','yr_renovated','city']


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final_features = np.array(int_features)
    dat_useen=pd.DataFrame([final_features],columns=cols)
    prediction = predict_model(model, data=dat_useen)
    prediction = int(prediction.Label[0])
    return render_template("home.html", prediction_text='The price of house is {}'.format(prediction))



if __name__ == '__main__':
    app.run(debug=True)
