#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 22:56:57 2020

@author: sahan
"""
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('kredi.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    
    current_job_dict = {"< 1 year" : 9 , "2 years" : 1 , "3 years" : 2 , "4 years" : 3 , "5 years" : 4 , "6 years" : 5 , "7 years" : 6 , "8 years" : 7 , "9 years" : 8 , "10+ years" : 0}
    home_ownershio_dict = {"Have Mortgage" : 0 , "Home Mortgage" : 1 , "Own Home" : 2, "Rent" : 3}
    purpose_dict = {"Business Loan" : 0 , "Buy House" : 1 , "Buy a Car" : 2 , "Debt Consolidation" : 3 , "Educational Expenses" : 4 , "Home Improvements" : 5,
                    "Medical Bills" : 6 , "Other" : 7 , "Take a Trip" : 8 , "Major-purchase" : 9 , "Moving" : 10 , "other" : 11 , "Renewable-energy" : 12 , "Small_business" : 13 ,
                    "Vacation" : 14 , "Wedding" : 15}
    
    data1 = request.form['a']
    data2 = request.form['b']
    if(data2 == "Short Term"):
        data2 = 1
    else:
        data2 = 0
    data3 = request.form['c']
    data4 = request.form['d']
    
    data5 = request.form['e']
    data5 = current_job_dict[data5]
    
    data6 = request.form['f']
    data6 = home_ownershio_dict[data6]
    
    data7 = request.form['g']
    data7 = purpose_dict[data7]
    
    data8 = request.form['h']
    data9 = request.form['i']
    data10 = request.form['j']
    data11 = request.form['k']
    
    arr = np.array([[data1, data2, data3, data4 , data5, data6, data7, data8, data9, data10, data11]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)



if __name__ == "__main__":
    app.run(host='localhost',port = 8000,debug=True)

