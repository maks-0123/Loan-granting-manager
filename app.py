import streamlit as st
import requests


st.title("Кредитная карта Maksimpremium")


with st.form("Заявка"):
    age = st.number_input("Возраст", 18, 100, 35)
    income = st.number_input("Доход", 0, 1000000, 100000)
    gender = st.selectbox("Пол", ["M", "F"])
    car = st.selectbox("Машина", ["Y", "N"])
    edu = st.selectbox("Образование", ["SCH", "UGR", "GRD"])
    rejects = st.number_input("Отказов")
    score = st.slider("Скоринг")
    
    if st.form_submit_button("Проверить"):
        data = {
            "age": age,
            "income": income,
            "gender_cd": gender,
            "car_own_flg": car,
            "education_cd": edu,
            "appl_rej_cnt": rejects,
            "Score_bki": score
        }
    
        r = requests.post("http://127.0.0.1:8000/score", json=data)
        if r.status_code == 200:
            result = r.json()
            if result['approved']:
                st.success("Ваш кредит одобрен спасибо, что выбрали нас!")
            else:
                st.error("К сожалению нам пришлось отказать вам в выдаче кредита,  но вы можете воспользоваться другими услугами нашего банка.")