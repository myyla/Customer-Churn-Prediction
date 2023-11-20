import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import json

# Charger les données depuis le fichier
file_path= "data\customer_churn.json"

# Lire les données JSON depuis le fichier
with open(file_path, 'r') as file:
    data_json = [json.loads(line) for line in file]

# Créer un DataFrame à partir des données JSON
data= pd.DataFrame(data_json)

print("\n                                                                      loading data ...")
print("###########################################################################################################################################################################################")
print(data)
print("###########################################################################################################################################################################################")

label_encoder = LabelEncoder()

# Appliquer le LabelEncoder 
data['Location'] = label_encoder.fit_transform(data['Location'])
data['Company'] = label_encoder.fit_transform(data['Company'])
data['Names'] = label_encoder.fit_transform(data['Names'])
data['Onboard_date'] = label_encoder.fit_transform(data['Onboard_date'])

print("                                                                       data preprocessing ... ")

# Diviser les données en features (X) et labels (y)
X = data.drop('Churn', axis=1)  
y = data['Churn']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser des données 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Initialiser les classificateurs
rf_classifier = RandomForestClassifier()
svm_classifier = SVC()
lr_classifier = LogisticRegression()

# Entraîner les classificateurs
rf_classifier.fit(X_train_scaled, y_train)
svm_classifier.fit(X_train_scaled, y_train)
lr_classifier.fit(X_train_scaled, y_train)

# Faire des prédictions sur les ensembles de test
rf_predictions = rf_classifier.predict(X_test_scaled)
svm_predictions = svm_classifier.predict(X_test_scaled)
lr_predictions = lr_classifier.predict(X_test_scaled)

print("                                                                       predicting with different classifier ... ")



# Évaluer les classificateurs
rf_accuracy = accuracy_score(y_test, rf_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)
lr_accuracy = accuracy_score(y_test, lr_predictions)

print("                                                                       comparing accuracy ... ")
print("###########################################################################################################################################################################################")
print(rf_accuracy)
print(svm_accuracy)
print(lr_accuracy)
print("###########################################################################################################################################################################################")

print("                                                                       selecting best classifier ... ")
# Sélectionner le meilleur classificateur
best_classifier = max([(rf_accuracy, 'Random Forest'), (svm_accuracy, 'SVM'), (lr_accuracy, 'Logistic Regression')])

print(f"                                                   The best classifier is {best_classifier[1]} with an accuracy of: {best_classifier[0]:.2f}\n")

# Sauvegarder le meilleur classificateur
if best_classifier[1] == 'Random Forest':
    joblib.dump(rf_classifier, 'RF_model.pkl')
elif best_classifier[1] == 'SVM':
    joblib.dump(svm_classifier, 'SVM_model.pkl')
else:
    joblib.dump(lr_classifier, 'LR_model.pkl')