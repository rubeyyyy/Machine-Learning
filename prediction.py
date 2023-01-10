import joblib

def predict(data):
    clf=joblib.load("outputModel/rf_model.sav")
    return clf.predict(data)