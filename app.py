import streamlit as st
import joblib
import pandas as pd

# Model ve Scaler yükle
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Kalp Sağlığı Tahmincisi")

# 1. Kullanıcıdan sadece istediğin 4 veriyi al
age = st.number_input("Yaşınız:", min_value=18, max_value=100, value=30)
sex = st.selectbox("Cinsiyet (0: Kadın, 1: Erkek):", [0, 1])
chol = st.number_input("Kolesterol Seviyesi:", min_value=100, max_value=600, value=200)
thal = st.selectbox("Thal Değeri (0, 1, 2, 3):", [0, 1, 2, 3])

# 2. Geri kalan sütunları modelin beklentisine göre "varsayılan" değerlerle doldur
# NOT: Bu isimler modelini eğitirken kullandığın sütun isimleriyle birebir aynı olmalı!
default_values = {
    'cp': 0, 'trestbps': 120, 'fbs': 0, 'restecg': 0,
    'thalach': 150, 'exang': 0, 'oldpeak': 0.0, 'slope': 1, 'ca': 0
}

if st.button("Tahmin Et"):
    # Sözlüğü birleştir
    input_dict = {
        'age': age, 'sex': sex, 'chol': chol, 'thal': thal,
        **default_values
    }
    
    # Sütun sırasını modelin eğitim sırasına göre diz
    # (Önemli: modelin hangi sırayla eğitildiyse o sırada listele!)
    feature_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_df = pd.DataFrame([input_dict])[feature_order]
    
    # Tahmin
    input_scaled = scaler.transform(input_df)
    tahmin = model.predict(input_scaled)
    
    if tahmin[0] == 1:
        st.error("Tahmin: Kalp hastalığı riski saptanmıştır. Lütfen bir uzmana danışın.")
    else:
        st.success("Tahmin: Kalp hastalığı riski düşük.")