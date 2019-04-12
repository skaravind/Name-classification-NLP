import pandas as pd 
import os

files = os.listdir()

df = pd.DataFrame(columns=['name', 'gender', 'race'])

for f in files:
	if 'csv' in f:
		if 'Indian' not in f:
			df1 = pd.read_csv(f)
			df1['name'] = df1['first name'] + df1['last name']
			del df1['first name'], df1['last name']
			cols = df1.columns.tolist()
			cols = cols[-1:] + cols[:-1]
			df1 = df1[cols]
			df = df.append(df1,ignore_index=True)
		else:
			df1 = pd.read_csv(f)
			df = df.append(df1,ignore_index=True)


df.to_csv('names_combined.csv', index=False)
