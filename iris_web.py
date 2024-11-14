#importar las librerias 
import pandas as pd
import pickle
import streamlit as st
#import streamlit_scrollable_textbox as stx


#extraer archivos picle

with open('lin_reg.pkl', 'rb') as li:
    lin_reg = pickle.load(li)

with open('log_reg.pkl', 'rb') as lo:
    log_reg = pickle.load(lo)
with open('svc_m.pkl', 'rb') as sv:
    svc_m = pickle.load(sv)

#metodo entrada 


#funcion para clasificar las plantas
def classify(num):
    if num == 0:
        return 'No tienes Diabetes'
    elif num == 1:
        return 'Tienes Diabetes'

def main():
    def parametros():
        
        
        age = st.sidebar.slider('age', 0, 120)
        bmi = st.sidebar.slider('bmi', 0, 100)
        HbA1c_level = st.sidebar.slider('HbA1c level', 1, 10)
        blood_glucose_level = st.sidebar.slider('blood glucose level', 70, 300)

        datos = {'age': float(age),
                 'bmi': float(bmi),
                 'HbA1c_level': float(HbA1c_level),
                 'blood_glucose_level': float(blood_glucose_level),
                 }
        features = pd.DataFrame(datos, index=[0])
        return features
   
    if 'menu' not in st.session_state:
        st.session_state.menu = False
    if not st.session_state.menu:
        st.title('Clasificación de diabetes')
        st.write(f'Descubre tu clasificación de riesgo de padecer diabetes de manera rápida y sencilla. Nuestra herramienta, basada en modelos de machine learning, te permite identificar el riesgo y clasificar tu situación según los resultado obtenidos.')
        st.write(f'Comienza con un simple test y obtén una clasificación personalizada junto con recomendaciones para mejorar tu calidad de vida y reducir el riesgo de diabetes')
        if st.button("Comenzar"):
            st.session_state.menu = True
    else:
        st.title(f'Test de clasificación')
        st.sidebar.header('Información')
        df = parametros()
        st.subheader('Información del usuario')
        st.write(df)
        if st.button('Checar'):
            prediction = log_reg.predict(df)[0]  # Obtén la predicción (suponiendo que es un solo valor)
    
            if prediction == 0:
             st.success(classify(prediction))  # Muestra en verde si es 0
            elif prediction == 1:
             st.error(classify(prediction))    # Muestra en rojo si es 1

            
                
           
            





    #st.sidebar.header('user  Input Parameters')
   

    
if __name__ == '__main__':
    main()