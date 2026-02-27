# https://platform.olimpiada-ai.ro/en/problems/184
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

raw_train = pd.read_csv("train.csv")
raw_test = pd.read_csv("test.csv")

# T1: Find the most frequent engine type
# This gives 0 points, can t find a fix
task1ans = str(raw_train['enginetype'].mode()[0]).strip().lower()

# T2: Average price of cars running on benzene
gasCars = raw_train[raw_train['fueltype'] == 'gas']
task2ans = gasCars['price'].mean()
task2ans = round(task2ans, 2)

# T3: Regression model predicting price

x_train = raw_train.drop(columns=['price', 'CarID', 'CarName'])
y_train = raw_train['price']

x_test = raw_test.drop(columns=['CarID', 'CarName'])

catCols = x_train.select_dtypes(include='object').columns
numCols = x_train.select_dtypes(exclude='object').columns

preprocessor = ColumnTransformer (
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), catCols),
        ('num', StandardScaler(), numCols)
    ]
)

param_grid = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [None, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
    "model__max_features": ["sqrt"]
}

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(x_train, y_train)
bestModel = grid_search.best_estimator_
predictions = bestModel.predict(x_test)

rows = []
rows.append({
    'subtaskID': 1,
    'datapointID': 1,
    'answer': task1ans
})
rows.append({
    'subtaskID': 2,
    'datapointID': 1,
    'answer': task2ans
})
for id,pred in zip(raw_test['CarID'], predictions):
  rows.append({
      'subtaskID': 3,
      'datapointID': id,
      'answer': pred
  })
subDf = pd.DataFrame(rows)
subDf.to_csv('submission.csv', index=False)
