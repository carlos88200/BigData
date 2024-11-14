#importar las librerias 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import pickle
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler



# Establecer el directorio donde quieres guardar los modelos
model_dir = "D:/Universidad/9 semestre/Big Data/AplicaciónWeb/"

# Asegurarse de que el directorio exista
os.makedirs(model_dir, exist_ok=True)

# Cargar los datos
diabetesuuno = pd.read_csv("D:/Universidad/9 semestre/Big Data/dataset/diabetes_prediction_dataset.csv")
diabetesuuno['blood_glucose_level'] = diabetesuuno['blood_glucose_level'].astype(float)


diabetesx= diabetesuuno[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
diabetesy = diabetesuuno['diabetes']

# Ver los valores de la columna 'diabetes'
print("valores")
print(diabetesuuno['diabetes'].values)

# Ver estadísticas descriptivas de la columna 'diabetes'
print("describe")
print(diabetesuuno['diabetes'].describe())
print("unique")
print(diabetesuuno['diabetes'].unique())



x = diabetesx#poner datos en variable dataframe
y = diabetesy #variable de entrenamkiento o objetiv7o

#aqui se van a separar los datos en entrenamiento y prueba 
x_train, x_test, y_train, y_test = train_test_split(x, y)

#scaler = StandardScaler()
#x_train_scaled = scaler.fit_transform(x_train)  # Escalar los datos de entrenamiento
#x_test_scaled = scaler.transform(x_test)  # Escalar los datos de prueba usando el mismo scaler



lin_reg = LinearRegression()
log_reg = LogisticRegression()

svc_m = SVC()
# aqui vamos a entrenar modelo

lin_regr = lin_reg.fit(x_train, y_train)
log_regr = log_reg.fit(x_train, y_train)
svc_mo=svc_m.fit(x_train, y_train)

y_pred = log_regr.predict(x_test)


# aqui se van aguarar datos en un archivo 
with open('lin_reg.pkl', 'wb') as li:
    pickle.dump(lin_regr, li)

with open('log_reg.pkl', 'wb') as lo:
    pickle.dump(log_reg, lo)

with open('svc_m.pkl', 'wb') as sv:
    pickle.dump(svc_m, sv)




