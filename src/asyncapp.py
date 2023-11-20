from flask import Flask, render_template, request, jsonify
import json
from kafka import KafkaProducer, KafkaConsumer
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import asyncio

app = Flask(__name__)

# Configure Kafka producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Configure Kafka consumer
consumer = KafkaConsumer('amal', bootstrap_servers='localhost:9092', group_id='group_id', value_deserializer=lambda x: json.loads(x.decode('utf-8')))

# Load pre-trained model
model = joblib.load('LR_model.pkl')

# Flag to indicate if a prediction is in progress
prediction_in_progress = False

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['POST'])
def predict():
    global prediction_in_progress

    if not prediction_in_progress:
        # Retrieve form data
        name = request.form['name']
        age = float(request.form['age'])
        total_Purchase = float(request.form['total_Purchase'])
        account_Manager = float(request.form['account_Manager'])
        years = float(request.form['years'])
        num_sites = float(request.form['num_sites'])
        onboard_date = request.form['onboard_date']
        location = request.form['location']
        company = request.form['company']

        # Create a dictionary with form data
        form_data = {
            'Names': name,
            'Age': age,
            'Total_Purchase': total_Purchase,
            'Account_Manager': account_Manager,
            'Years': years,
            'Num_Sites': num_sites,
            'Onboard_date': onboard_date,
            'Location': location,
            'Company': company
        }

        # Send data to Kafka topic
        producer.send('amal', value=form_data)

        prediction_in_progress = True

        # Wait a bit to let the Kafka consumer process the message
        asyncio.run(predict_async())

    return render_template('home.html', prediction=prediction)

# Asynchronous function to perform prediction
async def predict_async():
    global prediction_in_progress
    global model
    global prediction

    for message in consumer:
        preprocessed_data = pd.DataFrame([message.value])

        # Preprocess data
        label_encoder = LabelEncoder()
        preprocessed_data['Location'] = label_encoder.fit_transform(preprocessed_data['Location'])
        preprocessed_data['Company'] = label_encoder.fit_transform(preprocessed_data['Company'])
        preprocessed_data['Names'] = label_encoder.fit_transform(preprocessed_data['Names'])
        preprocessed_data['Onboard_date'] = label_encoder.fit_transform(preprocessed_data['Onboard_date'])

        # Make prediction
        prediction = model.predict(preprocessed_data)

        # Use the prediction as needed (e.g., print to console)
        print("Prediction:", prediction)

        prediction_in_progress = False
        break
    return render_template('home.html', prediction=prediction)

if __name__ == '__main__':
    # Start Flask application
    app.run(debug=True)