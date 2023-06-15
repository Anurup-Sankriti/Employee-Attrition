import pandas as pd
from collections import Counter
 # ['recency', 'no_licenses', 'no_certificates', 'Age', 'past_performance']
df = pd.read_csv('agent  attrition.csv')
'''print('Performance + recency:',df['past_performance'].corr(df['recency']))
print('Performance + license:',df['past_performance'].corr(df['no_licenses']))
print('Performance + certificates:',df['past_performance'].corr(df['no_certificates']))
print('Performance + age:',df['past_performance'].corr(df['Age']))'''

total_val = df['past_performance'].corr(df['recency']) + df['past_performance'].corr(df['no_licenses']) + df['past_performance'].corr(df['no_certificates'])
ratio_recency = df['past_performance'].corr(df['recency'])/total_val
ratio_license = df['past_performance'].corr(df['no_licenses'])/total_val
ratio_certificate = df['past_performance'].corr(df['no_certificates'])/total_val
# print(ratio_recency, ratio_license, ratio_certificate)
df['relationship'] = ratio_recency*df['recency'] + ratio_license*df['no_licenses']+ratio_certificate*df['no_certificates']
rel = list(df['relationship'])
rel.sort()
length = len(rel)
rel_small = []
for i in range(len(rel)):
    if i <= length/5 :
        rel_small.append(rel[i])
max_low = max(rel_small)

df['Attrition'] = df['relationship'].apply(lambda x: 0 if x >= max_low else 1)

#df.to_csv('relative_colm_added.csv', index=False)
