import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/', methods=["GET"])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    prob_prediction = model.predict_proba(final_features)
    output = prediction[0]
    output2 = round(np.amax(prob_prediction)*100, 2)
    output.capitalize()

    return render_template('index.html', prediction_text='Body fat percentage in a healthy range?: {}'.format(output.capitalize()) + ' with a {}'.format(output2) + '% probability')


@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
