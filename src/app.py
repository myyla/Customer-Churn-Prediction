from flask import Flask, render_template, request, redirect
import json
from kafka import KafkaProducer, KafkaConsumer
import pandas as pd
import joblib
from multiprocessing import Process
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Configurer le producteur Kafka
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Configurer le consommateur Kafka
consumer = KafkaConsumer('amal', bootstrap_servers='localhost:9092', group_id='group_id', value_deserializer=lambda x: json.loads(x.decode('utf-8')))

# Charger le modèle pré-entraîné
model = joblib.load('LR_modele.pkl')

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['POST'])
def predict():
    # Récupérer les données du formulaire
    name = request.form['name']
    age = float(request.form['age'])  # Assurez-vous que c'est un nombre
    total_Purchase = float(request.form['total_Purchase'])  # Assurez-vous que c'est un nombre
    account_Manager = float(request.form['account_Manager'])  # Assurez-vous que c'est un nombre
    years = float(request.form['years'])  # Assurez-vous que c'est un nombre
    num_sites = float(request.form['num_sites'])  # Assurez-vous que c'est un nombre
    onboard_date = request.form['onboard_date']
    location = request.form['location']
    company = request.form['company']

    # Créer un dictionnaire avec les données du formulaire
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

    # Envoyer les données au topic Kafka
    producer.send('amal', value=form_data)

    # # Lire les données prétraitées du topic Kafka
    for message in consumer:
        preprocessed_data = pd.DataFrame([message.value])

        # Prétraiter les données
        label_encoder = LabelEncoder()
        preprocessed_data['Location'] = label_encoder.fit_transform(preprocessed_data['Location'])
        preprocessed_data['Company'] = label_encoder.fit_transform(preprocessed_data['Company'])
        preprocessed_data['Names'] = label_encoder.fit_transform(preprocessed_data['Names'])
        preprocessed_data['Onboard_date'] = label_encoder.fit_transform(preprocessed_data['Onboard_date'])

        # Faire la prédiction
        prediction = model.predict(preprocessed_data)

        # Utiliser la prédiction comme bon vous semble, par exemple, l'afficher dans la console
        print("Prediction:", prediction)

        # Rediriger vers la page d'accueil après l'envoi des données
        #
        # Passer la prédiction au modèle pour qu'il l'affiche dans la page home.html
        return render_template('home.html', prediction=prediction)

# Démarrer le consommateur Kafka dans un processus séparé
def run_consumer():
    # Instantiate the TargetEncoder
    for message in consumer:
        preprocessed_data = pd.DataFrame([message.value])

        # # Prétraiter les données
        label_encoder = LabelEncoder()
        preprocessed_data['Location'] = label_encoder.fit_transform(preprocessed_data['Location'])
        preprocessed_data['Company'] = label_encoder.fit_transform(preprocessed_data['Company'])
        preprocessed_data['Names'] = label_encoder.fit_transform(preprocessed_data['Names'])
        preprocessed_data['Onboard_date'] = label_encoder.fit_transform(preprocessed_data['Onboard_date'])
        
        # Faire la prédiction
        prediction = model.predict(preprocessed_data)  # Assuming 'Name' is not used in training
        print("Prediction:", prediction)

# Démarrer le consommateur dans un processus séparé
consumer_process = Process(target=run_consumer)

if __name__ == '__main__':
    # Démarrer le consommateur Kafka
    consumer_process.start()

    # Démarrer l'application Flask
    app.run(debug=True)

    # Attendre que le consommateur se termine
    consumer_process.join()