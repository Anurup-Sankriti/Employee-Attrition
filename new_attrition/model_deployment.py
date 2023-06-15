import pickle
import pandas as pd
with open('model.pkl', 'rb') as file:
    loaded_pickle = pickle.load(file)
recency = input('Enter recency: ') # 0.08 - 28
no_licenses = input('Enter number of licences: ') # 1 - 8
no_certificates = input('Enter no. of certificates: ') # 0 - 4
Age = input('Enter your age: ') # 27 - 84, neg relation
past_performance=input('Enter past performance rating: ') # 1 - 51

data = {'recency': [recency], 'no_licenses':[no_licenses], 'no_certificates':[no_certificates], 'Age':[Age], 'past_performance':[past_performance]}
new_df = pd.DataFrame(data)

pred = loaded_pickle.predict(new_df)
if round(pred[0]) == 0:
    print('No attrition')
else:
    print('Attrition occurs')

