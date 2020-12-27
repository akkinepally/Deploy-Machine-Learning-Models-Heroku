from flask import Flask, jsonify, request, render_template, url_for, redirect
import pickle
import pandas as pd
import xgboost as xgb

app = Flask(__name__, template_folder='templates')

__model = None


def impute_nan_most_frequent_category(DataFrame, ColName):
    # .mode()[0] - gives first category name
    most_frequent_category = DataFrame[ColName].mode()[0]

    # replace nan values with most occured category
    DataFrame[ColName] = DataFrame[ColName]
    DataFrame[ColName].fillna(most_frequent_category, inplace=True)


def impute_nan_most_frequent_numerical(DataFrame, ColName):
    # .mode()[0] - gives first category name
    most_frequent_category = DataFrame[ColName].mean()

    # replace nan values with most occured category
    DataFrame[ColName] = DataFrame[ColName]
    DataFrame[ColName].fillna(most_frequent_category, inplace=True)


@app.route('/hello')
def hello():
    return "hi"


@app.route('/predict')
def index():
    return render_template('predict.html')


@app.route('/predicted', methods=['GET', 'POST'])
def predict_loan_predictions():
    uploaded_file = request.files['file']
    test_data = pd.read_csv(uploaded_file)
    for Columns in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
        impute_nan_most_frequent_category(test_data, Columns)
    for Columns in ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
        impute_nan_most_frequent_numerical(test_data, Columns)
    test_data['Dependents'] = test_data['Dependents'].replace('3+', '3')
    test_data['Dependents'] = test_data['Dependents'].astype(str).astype(float)
    test_data = test_data.drop(['Loan_ID', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    objlist = test_data.select_dtypes(include="object").columns
    for feat in objlist:
        test_data[feat] = le.fit_transform(test_data[feat].astype(str))
    print(test_data.info())
    predicted_output = __model.predict(test_data)
    print(type(predicted_output))
    predicted_output_json = jsonify({'predicted_output': predicted_output.tolist()})
    # predicted_output_json.headers.add('Access-Control-Allow-Origin', '*')
    return predicted_output_json


def load_model():
    global __model
    with open('loan_prediction.pickle', 'rb') as f:
        __model = pickle.load(f)
    return __model

"""
@app.route('/upload')
def upload_file():
    return render_template('predict.html')


from werkzeug.utils import secure_filename


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file1():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
    return 'file uploaded successfully'
"""

if __name__ == "__main__":
    print('starting flask app')
    load_model()
    app.run(debug=True)
