import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("cleaned_churn.csv")

X = data.drop(['Churn'], axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

transformer = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(sparse_output=False, dtype=int, drop=None), 
     ['InternetService', 'Contract', 'PaymentMethod']),
], remainder='passthrough')

X_train_processed = transformer.fit_transform(X_train)
X_test_processed = transformer.transform(X_test)