import pickle
import pandas as pd
with open('model.pkl', 'rb') as file:
    loaded_pickle = pickle.load(file)
recency = float(input('Enter recency: '))
no_licenses = float(input('Enter number of licences: '))
no_certificates = float(input('Enter no. of certificates: '))
past_performance=float(input('Enter past performance rating: '))

if recency > 8.8: recency = 8.8
if no_licenses > 7: no_licenses = 7
if no_certificates > 3: no_certificates = 3
if past_performance > 14.375: past_performance = 14.375

data = {'recency': [recency], 'no_licenses':[no_licenses], 'no_certificates':[no_certificates], 'past_performance':[past_performance]}
new_df = pd.DataFrame(data)

pred = loaded_pickle.predict(new_df)
if round(pred[0]) == 0:
    print('No attrition')
else:
    print('Attrition occurs')
