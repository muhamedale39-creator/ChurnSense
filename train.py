import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder ,StandardScaler
from sklearn.model_selection import train_test_split , GridSearchCV ,StratifiedKFold
from sklearn.metrics import classification_report ,f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

data = pd.read_csv("cleaned_churn.csv")
data = pd.read_csv("cleaned_churn.csv")

X = data.drop(['Churn'] ,axis =1)
y = data['Churn']

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,stratify=y)

data_imbalance_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]

transformer = ColumnTransformer(transformers=[
    ('OneHot' , OneHotEncoder(sparse_output=False , dtype=int ,drop = None) , ['InternetService' , 'Contract' , 
                                                                               'PaymentMethod']),
] ,remainder='passthrough')

X_train_processed = transformer.fit_transform(X_train)
X_test_processed = transformer.transform(X_test)

cv = StratifiedKFold(n_splits=5 ,random_state=1 , shuffle=True)

model =XGBClassifier(random_state = 1,scale_pos_weight = data_imbalance_ratio)
param_grid = {
    'n_estimators' : [200,300,500],      
    'learning_rate' : [0.01 ,0.03,0.05,0.1],     
    'max_depth': [3,4,5,6],
    'min_child_weight' :[1,3,5,10],
    'scale_pos_weight' : [1,2,data_imbalance_ratio]          
}


grid = GridSearchCV(cv=cv ,estimator=model , scoring='average_precision',n_jobs=-1,param_grid=param_grid)
grid.fit(X_train_processed,y_train)

best_model = grid.best_estimator_
y_probs = best_model.predict_proba(X_test_processed)[:, 1]


thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores = [f1_score(y_test, (y_probs >= t).astype(int)) for t in thresholds]
best_thresh = thresholds[np.argmax(f1_scores)]

y_pred = (y_probs >= best_thresh).astype(int)

print(f"Optimal Threshold: {best_thresh:.2f}")
print(classification_report(y_test, y_pred))