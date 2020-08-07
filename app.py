import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle



app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['ApprovalFY', 'Term', 'NoEmp','NewExist', 'CreateJob',
                     'RetainedJob', 'FranchiseCode', 'UrbanRural',
                     'RevLineCr']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 1:
        res_val = "** Non-Default **"
    else:
        res_val = "** Default **"
        

    return render_template('index.html', prediction_text='The Loan Borrower may {}'.format(res_val))


if __name__ == "__main__":
    app.run(debug=True)
