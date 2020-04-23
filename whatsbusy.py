import pickle
import numpy as np

def get():
    mean, std = pickle.load(open('mean_std.pkl', 'rb'))
    latest = pickle.load(open('xtest.pkl', 'rb'))
    return mean, std, latest

def predict_next_day(model='rf'):
    mean, std, xtest = get()
    
    model += '.pkl'
    reg = pickle.load(open(model, 'rb'))
    pred = reg.predict(np.array(xtest[-1]).reshape(1, -1))
    pred *= std
    pred += mean
    
    print('Prediction for next day: ', pred[0])
