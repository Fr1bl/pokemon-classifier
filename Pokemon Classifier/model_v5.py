import pandas as pd #для работы с таблицами (DataFrame)
import pickle
from sklearn.ensemble import RandomForestClassifier  #алгоритм случайного леса для классификации
from sklearn.model_selection import train_test_split  #чтобы разделить данные на обучение и тест
from sklearn.metrics import classification_report  #чтобы получить метрики качества классификации
from sklearn.impute import SimpleImputer  #импортирую класс, с помощью которого буду заполнять пропущенные значения в числовых данных
from sklearn.preprocessing import LabelEncoder  #импортирую класс для преобразования категориальных данных в числовые метки
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer  #импортирую класс, чтобы объединить разные типы преобразований для разных столбцов
from sklearn.pipeline import Pipeline  #импортирую класс, чтобы выстроить последовательность шагов обработки данных

#загрузка датасета
df = pd.read_csv("pokedex_(Update_04.21).csv")

if 'status' in df.columns:
    #если 'status' есть, просто копирую его значения в 'Class'
    #предполагаю, что значения уже текстовые, например: 'Common', 'Rare', 'Legendary'
    df['Class'] = df['status']

numeric_features = [
    'total_points', 'hp', 'attack', 'defense', 
    'sp_attack', 'sp_defense', 'speed', 'catch_rate',
    'generation', 'type_number', 'base_experience',
    'abilities_number', 'egg_cycles'
]  #создаю список числовых признаков, которые содержат количественные значения — такие как очки, здоровье, атака, поколение и т.д., их нужно будет обработать числовыми методами

categorical_features = [
    'growth_rate', 'ability_hidden'
]  #создаю список категориальных признаков, в которых значения представлены в виде строк или категорий, например способности или скорость роста — их нужно будет закодировать

numeric_features = [col for col in numeric_features if col in df.columns]  #фильтрую список числовых признаков, оставляя только те, которые действительно есть в датафрейме df, чтобы избежать ошибок при дальнейшем использовании
categorical_features = [col for col in categorical_features if col in df.columns]  #аналогично фильтрую список категориальных признаков — оставляю только существующие в df, чтобы избежать проблем при трансформации

print("Числовые признаки:", numeric_features)  #вывожу на экран отфильтрованные числовые признаки, чтобы убедиться, что всё корректно
print("Категориальные признаки:", categorical_features)  #вывожу на экран отфильтрованные категориальные признаки, чтобы проверить корректность списка

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])  #создаю пайплайн обработки числовых признаков, где на первом этапе заменяю пропущенные значения на медиану по столбцу — медиана устойчива к выбросам

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])  #создаю пайплайн обработки категориальных признаков: сначала заполняю пропущенные значения наиболее часто встречающимся, затем применяю one-hot кодирование, при этом игнорирую неизвестные значения на этапе предсказания

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])  #создаю трансформер, который применяет соответствующую обработку к разным типам признаков — числовым и категориальным, объединяя их в единый обработчик


#проверяю наличие целевого столбца
if 'Class' not in df.columns:
    raise ValueError("Столбец 'Class' не создан")

#разделяю данные на обучающую и тестовую выборки:
#test_size=0.2 → 20% данных пойдут в тестовый набор
#random_state=42 → фиксируем случайное разбиение для воспроизводимости
X = df[numeric_features + categorical_features]
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#RandomForestClassifier — это алгоритм "Случайный лес".
#Он строит много "деревьев решений" (в данном случае 200 деревьев) и голосует, чтобы принять финальное решение.
#Каждое дерево "спрашивает" что-то вроде:
#Атака > 120? → да → дальше
#Скорость < 70? → да → дальше
#и в итоге решает: "легендарный" или "нет".
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    ))
])
model.fit(X_train, y_train)

#вывожу метрики: точность, полнота и др.
#classification_report показывает:
#precision — насколько часто предсказания "легендарный" были верными.
#recall — сколько настоящих легендарных покемонов модель смогла "поймать".
#f1-score — среднее между precision и recall.
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

#сохранение модели
with open('model_v5.pkl', 'wb') as f:
    pickle.dump(model, f)