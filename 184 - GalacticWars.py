# https://platform.olimpiada-ai.ro/en/problems/184
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
import pandas as pd

train_raw = pd.read_csv('train_galactic_wars.csv')
test_raw = pd.read_csv('test_galactic_wars.csv')

# T1
task1ans = ((test_raw['fight_planet'] == 'Nabira') & (test_raw['weather_conditions'] == 'Snow')).sum()
# T2
task2ans = (train_raw['weapon_calmtrooper'] == 'Experimental Weapon').sum()

# Helper functions for T3 and T4
x_train = train_raw.drop(columns=['FightID', 'winner'])
y_train = train_raw['winner']
x_test = test_raw.drop(columns='FightID')

# Select percentages so we can later turn them into integers
percentCols = ['armour_jedai', 'armour_calmtrooper']
# So we don t onehot percentages we remove them from the list and add them to the float one
catCols = x_train.select_dtypes(include='object').columns
catCols = catCols.difference(percentCols)
boolCols = x_train.select_dtypes(include='bool').columns
floatCols = x_train.select_dtypes(include='float').columns
floatCols = floatCols.tolist() + percentCols

def cleanValues(df):
  df = df.copy()
  # fix weapon
  df['weapon_calmtrooper'] = df['weapon_calmtrooper'].replace('Experimental Weapon','Double Blaster')
  # bool -> int
  df[boolCols] = df[boolCols].astype(int)
  # % -> int
  for col in percentCols:
    df[col] = df[col].str.replace('%', '', regex=False).astype(float)

  return df

preprocessor = ColumnTransformer(
  transformers = [
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), catCols),
    ('num', StandardScaler(), floatCols)
  ]
)

# T3

pipeline1 = Pipeline([
    ('clean', FunctionTransformer(cleanValues)),
    ('preprocess', preprocessor),
    ('model', KMeans(random_state=42, n_clusters=3, n_init=20))
    ])
pipeline1.fit(x_train)
predT3 = pipeline1.predict(x_test)

# T4

pipeline2 = Pipeline([
    ('clean', FunctionTransformer(cleanValues)),
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

param_grid = {
    'model__n_estimators': [150, 250],
    'model__max_depth': [None, 8, 15],
    'model__min_samples_leaf': [1, 3],
}

gridSearch = GridSearchCV(
    pipeline2,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs = -1
)

gridSearch.fit(x_train, y_train)
model = gridSearch.best_estimator_
predT4 = model.predict(x_test)

# DF
rows = []
rows.append({
    "subtaskID": 1,
    "datapointID": 1,
    "answer": task1ans
})
rows.append({
    "subtaskID": 2,
    "datapointID": 2,
    "answer": task2ans
})
for id,pred in zip(test_raw['FightID'], predT3):
  rows.append({
    "subtaskID": 3,
    "datapointID": id,
    "answer": pred
  })
for id,pred in zip(test_raw['FightID'], predT4):
  rows.append({
    "subtaskID": 4,
    "datapointID": id,
    "answer": pred
  })
subDf = pd.DataFrame(rows)
subDf.to_csv('submission.csv', index=False)
