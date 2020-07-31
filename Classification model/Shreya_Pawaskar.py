import pandas as pd
import sys

dataset = pd.read_csv(sys.argv[1])

df = pd.DataFrame(dataset)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['output'] = labelencoder.fit_transform(df['Label'])
df['year1'] = labelencoder.fit_transform(df['Which-year are you studying in?'])

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

x4=df[["CGPA/ percentage","Rate your written communication skills [1-10]","Rate your verbal communication skills [1-10]","year1"]]
y4=df['output']

x_train1, x_test1, y_train1,y_test1=train_test_split(x4,y4,test_size=0.20,random_state=42)

ans3=RandomForestClassifier()
ans3.fit(x_train1, y_train1)
y_pred4= ans3.predict(x_test1)
print(f1_score(y_test1, y_pred4))
