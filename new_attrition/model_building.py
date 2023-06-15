import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pickle
df = pd.read_csv('relative_colm_added.csv')
df = df.drop('relationship', axis = 1)
X = df.drop('Attrition', axis = 1)
Y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)
print('r2:', r2)
print('mse: ', mse)

pickle.dump(model, open('model.pkl', 'wb'))