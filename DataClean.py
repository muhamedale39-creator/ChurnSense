import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("dataset.csv")

data = data.drop( "customerID", axis=1)

sns.barplot(x='InternetService', y='Churn', data=data)
plt.title('Churn Rate by Internet Service')
plt.show()

sns.boxplot(x='Contract', y='tenure', hue='Churn', data=data)
plt.title('Tenure Distribution by Contract and Churn')
plt.show()

data['gender'] = data["gender"].map({
 'Male' : 1,
 'Female' :0
})
data.loc[data['MultipleLines'] == 'No phone service' , 'MultipleLines'] = 'No'

Columns = ['OnlineSecurity' , 'OnlineBackup' , 'DeviceProtection' , 'TechSupport' , 'StreamingTV' ,'StreamingMovies']

for col in Columns:
    data.loc[data[col]== 'No internet service' , col] = 'No'

Col = [ 'Partner' , 'Dependents' , 'PhoneService' , 'OnlineSecurity' , 'OnlineBackup' ,'Churn' ,
     'DeviceProtection' , 'TechSupport' , 'StreamingTV' ,'StreamingMovies' ,'MultipleLines','PaperlessBilling']

for col in Col:
    data[col] = data[col].map({
    'Yes' : 1,
    'No' : 0
 })
    
data['TotalCharges'] = data['TotalCharges'].str.replace("r(^\d' ')" , '' , regex=True)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'] , errors='coerce' )
data['TotalCharges'] = data['TotalCharges'].fillna(0)

print(data.info())
data.to_csv("cleaned_churn.csv" , index=False)
print(data['Churn'].value_counts())




