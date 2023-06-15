import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


df = pd.read_csv('relative_colm_added.csv')
df = df.drop('relationship', axis = 1)
df = df.drop('Age', axis = 1)

z_scores = np.abs(stats.zscore(df))

threshold = 3
outlier_indices = np.where(z_scores > threshold)
df_cleaned= df.drop(outlier_indices[0])
print(df.shape)
print(df_cleaned.shape)
has_null = df_cleaned.isnull().values.any()

# Print the result
print(min(list(df_cleaned['recency'])), max(list(df_cleaned['recency'])))
print(min(list(df_cleaned['no_licenses'])), max(list(df_cleaned['no_licenses'])))
print(min(list(df_cleaned['no_certificates'])), max(list(df_cleaned['no_certificates'])))
print(min(list(df_cleaned['past_performance'])), max(list(df_cleaned['past_performance'])))

X = df_cleaned.drop('Attrition', axis = 1)
Y = df_cleaned['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)
pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)
print('mse', mse)
print('r2', r2)

#pickle.dump(model, open('model.pkl', 'wb'))

