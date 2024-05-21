import pmdarima as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

train = pd.read_csv(appl.csv)
def errors(prediction,actual):
    
    mae = np.mean(np.abs(prediction - actual))
    
    mape = np.mean(np.abs(prediction - actual)/np.abs(actual))
    
    rmse = np.mean((prediction - actual)**2)**0.5
    
    return({'mae':mae,'mape':mape,'rmse':rmse})

model = pm.auto_arima(train, start_p=1, start_q=1,
                      test='adf',       
# use adftest to find optimal 'd'
                      max_p=3, max_q=3, 
# maximum p=3 and q=3
                      m=12,              
# periodicity of 12 months as the data timeline is in months
                      d=None,           
# let the model determine 'd'
                      seasonal=True,   # Seasonality
                      start_P=0, 
                      D=1, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
print(model.summary())