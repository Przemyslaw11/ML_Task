''' 6. Create a very simple REST API that will serve your models '''
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the model choice from the user input
    model_choice = request.json['model_choice']
    
    # Load the chosen model
    if model_choice == 'heuristic':
        model = joblib.load('heur_model.pkl')
    elif model_choice == 'dt_clf':
        model = joblib.load('dt_clf_model.pkl')
    elif model_choice == 'rf_clf':
        model = joblib.load('rf_clf_model.pkl')
    elif model_choice == 'neural_network':
        model = joblib.load('NN_model.pkl')
    else:
        return jsonify({'error': 'Invalid model choice'})
    
    # Get the necessary input features from the user input
    input_features = request.json['input_features']
    
    # Make the prediction
    prediction = model.predict(input_features)
    
    # Return the prediction
    return jsonify({'prediction': prediction.tolist()})
