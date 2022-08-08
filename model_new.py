import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from statsmodels.tsa.stattools import adfuller
import statsmodels as sm
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as smapi


class File_Pass:
    global lst_output
    def __init__(self,name,final_features):
        self.name = name
        self.length = final_features[0][0]
        #self.output_len = final_features[0][1]

    #Converting words to integer values
    def convert_to_int(word):
        word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                    'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
        return word_dict[word]

    
    def Analysize(self):
        #Ho: It is non stationary
        #H1: It is stationary

        def adfuller_test(value):
            result=adfuller(value)
            labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
            for value,label in zip(result,labels):
                print(label+' : '+str(value) )
            if result[1] <= 0.05:
                print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
            else:
                print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
    
        df = pd.read_csv('dataset/'+self.name)
        adfuller_test(df['high'])
        df['high First Difference'] = df['high'] - df['high'].shift(1)
        df['high'].shift(1)
        ## Again test dickey fuller test
        adfuller_test(df['high First Difference'].dropna())
        df['Seasonal First Difference']=df['high']-df['high'].shift(self.length)
        ## Again test dickey fuller test
        adfuller_test(df['Seasonal First Difference'].dropna())
        
        model = smapi.tsa.arima.ARIMA(df['high'], order=(1,1,1))
        result = model.fit()

        df['forecast']=result.predict()
        df[['low','forecast']].plot(figsize=(12,8))

        #pickle.dump(model, open('model.pkl','wb'))
        # Loading model to compare the results
        
        #model = pickle.load(open('model.pkl','rb'))
        
        # demonstrate prediction for next 10 days
       
        plt.savefig('static\\result.png')
        

'''
fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        fig1.savefig('static\\my_plot.png')
        #plt.show()'''
        

