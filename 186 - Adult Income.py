# https://platform.olimpiada-ai.ro/en/problems/186

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

train_raw = pd.read_csv('train.csv')
test_raw = pd.read_csv('test.csv')

# T1
filter1 = train_raw[(train_raw['income'] == '>50K') & (train_raw['native_country'] != 'United-States')]
task1ans = filter1['native_country'].value_counts().idxmax()

# T2
train_raw['is_high_income'] = (train_raw['income'] == '>50K').astype(bool)
rates = train_raw.groupby('occupation')['is_high_income'].mean()
task2ans = rates.idxmax()
train_raw = train_raw.drop(columns='is_high_income')

# T3

x_train = train_raw.drop(columns='sampleid')
y_train = train_raw['income']
x_test = test_raw.drop(columns='sampleid')

colsToDrop = ['education', 'profile_description', 'fnlwgt']
numericCols = ['age', 'educational_num', 'capital_gain', 'capital_loss', 'hours_per_week']
oneHotCols = ['workclass', 'marital_status','relationship', 'gender', 'race']
targetCols = ['occupation', 'native_country']

x_train = x_train.drop(columns=colsToDrop)
x_test = x_test.drop(columns=colsToDrop)

le = LabelEncoder()
y_train = le.fit_transform(y_train)

def targetEncode(df):
  pass

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numericCols),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), oneHotCols),
        ('target', TargetEncoder(), targetCols)
    ]
)

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', XGBClassifier())
])

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.1, 0.05],
}

gridSearch = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)
gridSearch.fit(x_train, y_train)
model = gridSearch.best_estimator_
predictions = model.predict(x_test)
predictions = le.inverse_transform(predictions)

# T4 ?
# DF

rows = []
rows.append({
    'subtaskID': 1,
    'datapointID': 1,
    'answer': task1ans
})
rows.append({
    'subtaskID': 2,
    'datapointID': 2,
    'answer': task2ans
})
for id,pred in zip(test_raw['sampleid'], predictions):
  rows.append({
    'subtaskID': 3,
    'datapointID': id,
    'answer': pred
  })
subDf = pd.DataFrame(rows)
subDf.to_csv('submission.csv', index=False)
