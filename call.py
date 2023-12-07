import requests

def make_prediction(input_data):
    url = 'http://127.0.0.1:5000/predict'
    try:
        # Send a POST request to the Flask server
        response = requests.post(url, json={'features': input_data})

        # Check the response status code
        if response.status_code == 200:
            result = response.json()
            prediction = result.get('prediction')
            if prediction is not None:
                # Mapping numeric predictions to species names
                species_mapping = {
                    0: 'Setosa',
                    1: 'Versicolor',
                    2: 'Virginica'
                }
                # Return the corresponding species name if available
                return species_mapping.get(prediction, "Unknown species")
            else:
                return "Prediction not found in the response."
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"An error occurred: {e}"

# Example input data for prediction
input_data = [6.3,3.3,4.7,1.6]

# Make a prediction
prediction_result = make_prediction(input_data)
print(f"Prediction: {prediction_result}")
