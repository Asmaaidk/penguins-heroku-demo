from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
filename = 'model.pkl'
model = pickle.load(open(filename, 'rb'))

@app.route('/')
def home():
	return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
	features = [int(x) for x in request.form.values()]
	final_features = [np.array(features)]
	prediction =  model.predict(final_features)

	output = prediction
	return render_template('index.html', prediction_text = 'The predectied flipper length is {} mm'.format(output))

if __name__ == "__main__":
	app.run(port=5000, debug = True)