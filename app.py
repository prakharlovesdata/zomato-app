from flask import Flask, abort, jsonify, request, render_template
from sklearn.externals import joblib

import json
import pandas as pd
import numpy as np
# load the built-in model
gbr = joblib.load('km5.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('app.html')

@app.route('/app',methods=['POST'])
def get_delay():

    result=request.form
    Average_Cost  = result['Average Cost for two']
    Price_range = result['Price range']
    Aggregate_rating = result['Aggregate rating']
    votes = result['Votes']
    Rating_no = result['Rating_no']
    # we create a json object that will hold data from user inputs
    #user_input = {'Average Cost for two':Average_Cost,'Price range':Price_range, 'Aggregate_rating':Aggregate_rating, 'Votes':votes, 'Rating_no':Rating_no}

    df = pd.DataFrame(columns=['Average Cost for two', 'Price range', 'Aggregate rating', 'Votes','Rating_no'], index=['0'])
    df.loc['0'] = pd.Series({'Average Cost for two':Average_Cost,'Price range':Price_range, 'Aggregate rating':Aggregate_rating, 'Votes':votes, 'Rating_no':Rating_no})
    # get the price prediction
    price_pred = gbr.predict(df)
    price_pred = np.exp(price_pred)





    #return a json value
    return json.dumps({'value': round(price_pred[0],2)});

if __name__ == '__main__':
    app.run(debug=True)
