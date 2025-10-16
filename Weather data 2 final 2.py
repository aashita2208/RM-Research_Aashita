import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\aashi\Downloads\weather_classification_data.csv")
df1 = df.copy()
df1.dropna(inplace=True)
X_df = df1.iloc[:, :-1] 
y = df1.iloc[:, -1].values 
X_encoded = pd.get_dummies(X_df)  
X = X_encoded.values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
mm = MinMaxScaler()
X_train[:, :] = mm.fit_transform(X_train[:, :])
X_test[:, :] = mm.transform(X_test[:, :])
sta = StandardScaler()
X_train[:, :] = sta.fit_transform(X_train[:, :])
X_test[:, :] = sta.transform(X_test[:, :])
processed_df = pd.DataFrame(X, columns=X_encoded.columns)
processed_df['Weather Type'] = y
processed_df.to_csv(r"C:\Users\aashi\Downloads\weather_processed.csv", index=False)
print("Processed dataset saved")
