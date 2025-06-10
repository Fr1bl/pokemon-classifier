import streamlit as st
import pandas as pd
import pickle

st.title("Классификатор покемонов")
st.markdown("Настройте параметры покемона и получите предсказание его класса")

@st.cache_resource
def load_model():
    with open("model_v5.pkl", 'rb') as f:
        return pickle.load(f)

model = load_model()

#боковая панель для включения/отключения признаков
with st.sidebar:
    st.header("Настройки признаков")
    use_total_points = st.checkbox("Использовать Total Points", value=True)
    use_catch_rate = st.checkbox("Использовать Catch Rate", value=True)
    use_generation = st.checkbox("Использовать Generation", value=True)
    use_type = st.checkbox("Использовать Type Number", value=True)
    use_growth = st.checkbox("Использовать Growth Rate", value=False)
    use_exp = st.checkbox("Использовать Base Experience", value=True)
    use_egg_cycles = st.checkbox("Использовать Egg Cycles", value=False)
    use_abilities = st.checkbox("Использовать Abilities", value=False)
    use_hidden_ability = st.checkbox("Использовать Hidden Ability", value=False)
#основные параметры
st.header("Основные характеристики")
col1, col2, col3 = st.columns(3)
with col1:
    hp = st.slider("HP", 1, 200, 80)
    attack = st.slider("Attack", 1, 200, 100)
    defense = st.slider("Defense", 1, 200, 70)
with col2:
    sp_atk = st.slider("Sp. Atk", 1, 200, 60)
    sp_def = st.slider("Sp. Def", 1, 200, 90)
    speed = st.slider("Speed", 1, 200, 110)
with col3:
    if use_total_points:
        total_points = st.slider("Total Points", 175, 800, 500)
    if use_catch_rate:
        catch_rate = st.slider("Catch Rate", 3, 255, 45)

#дополнительные параметры
if use_generation or use_type or use_exp or use_egg_cycles:
    st.header("Дополнительные параметры")
    cols = st.columns(6)
    if use_generation:
        with cols[0]:
            generation = st.slider("Generation", 1, 8, 1)
    if use_type:
        with cols[1]:
            type_number = st.slider("Type Number", 1, 2, 1)
    if use_exp:
        with cols[2]:
            base_experience = st.slider("Base Experience", 50, 600, 250)
    if use_egg_cycles:
        with cols[3]:
            egg_cycles = st.slider("Egg Cycles", 5, 120, 20)
    if use_abilities:
        with cols[4]:
            abilities_number = st.slider("Abilities Number", 1, 3, 1)
    if use_hidden_ability:
        with cols[5]:
            ability_hidden = st.checkbox("Hidden Ability", value=False)
#Growth rate (категориальный параметр)
if use_growth:
    growth_rate = st.selectbox("Growth Rate", 
                             ["Medium Slow", "Slow", "Medium Fast", "Fast", "Erratic", "Fluctuating"])


if st.button("Определить класс покемона"):
    #собираю все включенные параметры
    input_data = {
        'hp': hp,
        'attack': attack,
        'defense': defense,
        'sp_attack': sp_atk,
        'sp_defense': sp_def,
        'speed': speed
    }
    
    if use_total_points:
        input_data['total_points'] = total_points
    if use_catch_rate:
        input_data['catch_rate'] = catch_rate
    if use_generation:
        input_data['generation'] = generation
    if use_type:
        input_data['type_number'] = type_number
    if use_exp:
        input_data['base_experience'] = base_experience
    if use_growth:
        input_data['growth_rate'] = growth_rate
    if use_egg_cycles:
        input_data['egg_cycles'] = egg_cycles
    if use_abilities:
        input_data['abilities_number'] = abilities_number
    if use_hidden_ability:
        input_data['ability_hidden'] = ability_hidden
    input_df = pd.DataFrame([input_data])
    
    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        
        st.success(f"### Результат: {prediction}")
        
        with st.expander("Подробности предсказания"):
            st.write(f"Вероятности классов:")
            for i, cls in enumerate(model.classes_):
                st.write(f"- {cls}: {proba[i]*100:.1f}%")
            
            if hasattr(model, 'feature_importances_'):
                st.write("\nВажные признаки:")
                features = input_df.columns
                importances = model.feature_importances_
                for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
                    st.write(f"- {feat}: {imp:.3f}")
    
    except Exception as e:
        st.error(f"Ошибка предсказания: {str(e)}")
        st.write("Используемые параметры:", input_data)