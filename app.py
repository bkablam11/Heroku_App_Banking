import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('../Deployment-flask/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    if output == 1:
        return render_template('Answers1.html', prediction_text = 'Le client va se désabonner')
    else:
        return render_template('Answers0.html', prediction_text='Le client ne va pas se désabonner')


if __name__ == "__main__":
    app.run(debug=True)