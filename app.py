import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get features from form
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    
    # Make prediction
    prediction = model.predict(features)

    # Round and format the output
    output = round(prediction[0], 2)

    # Return result with prediction_text
    return render_template('index.html', prediction_text=f'Predicted CO2 Emission: {output}%')

if __name__ == "__main__":
    app.run(debug=True)
