import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

#random seed
seed=42

#read original dataset
iris_df=pd.read_csv('/home/rubi/Machine-Learning/Data/Iris.csv')
iris_df= iris_df.sample(frac=1, random_state=seed)

#selecting features and target
X=iris_df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y=iris_df[['Species']]


#split

X_train,X_test, y_train, y_test= train_test_split(X,y, test_size=0.3, random_state=seed, stratify=y)

#create an instanxe of random forest classifier

clf= RandomForestClassifier(n_estimators=100)

#train classifier on training data
clf.fit(X_train, y_train)

#predict on the test
y_pred= clf.predict(X_test)

#calculate accuracy
accuracy=accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')  #aCCURACY:0.91


#save the model.disk
joblib.dump(clf,"outputModel/rf_model.sav")