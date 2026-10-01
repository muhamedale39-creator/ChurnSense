import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold 
from sklearn.metrics import classification_report, f1_score 
from xgboost import XGBClassifier 

data = pd.read_csv("cleaned_churn.csv") 

X = data.drop(['Churn'],axis=1) 
y = data['Churn'] 

X_train,X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
counts = y_train.value_counts()

data_imbalance_ratio =counts.get(0,counts.get('No',1)) /counts.get(1,counts.get('Yes',1))

categorical_cols =['InternetService','Contract','PaymentMethod']

transformer =ColumnTransformer(transformers=[ 
    ('OneHot',OneHotEncoder(sparse_output=False, drop=None),categorical_cols), 
],remainder='passthrough') 

X_train_processed =transformer.fit_transform(X_train) 
X_test_processed =transformer.transform(X_test) 

cv =StratifiedKFold(n_splits=5, random_state=1, shuffle=True) 

model =XGBClassifier(random_state=1, scale_pos_weight=data_imbalance_ratio, n_jobs=1) 

param_grid= { 
    'n_estimators': [100, 200, 300], 
    'learning_rate': [0.03, 0.1], 
    'max_depth': [3, 4, 5], 
    'min_child_weight': [1, 3, 5], 
    'scale_pos_weight': [1, data_imbalance_ratio] 
} 

grid =GridSearchCV(cv=cv,estimator=model,scoring='average_precision',n_jobs=-1,param_grid=param_grid) 
grid.fit(X_train_processed,y_train) 

best_model =grid.best_estimator_ 
y_probs =best_model.predict_proba(X_test_processed)[:, 1] 

thresholds =np.arange(0.1, 0.9, 0.05) 
f1_scores =[f1_score(y_test, (y_probs >= t).astype(int), pos_label=1) for t in thresholds] 
best_thresh =thresholds[np.argmax(f1_scores)] 

y_pred =(y_probs>=best_thresh).astype(int) 

print(classification_report(y_test, y_pred))
