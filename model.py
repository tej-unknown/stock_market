import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from keras.layers import Dropout
import preprocessor as p
import csv
#from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from numpy import array

class File_Pass:
    global lst_output
    def __init__(self,name,final_features):
        self.name = name
        self.length = final_features[0][0]
        #self.output_len = final_features[0][1]

        
    def lstm_algo(self):
        da = self.name
        def prepare_data(timeseries_data, n_features):
            X, y =[],[]
            for i in range(len(timeseries_data)):
                # find the end of this pattern
                end_ix = i + n_features
                # check if we are beyond the sequence
                if end_ix > len(timeseries_data)-1:
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
                X.append(seq_x)
                y.append(seq_y)
            return np.array(X), np.array(y)

        df_train = pd.read_csv('dataset/'+self.name)
        training_set = df_train.iloc[:, 2].values
        #print(training_set,'\n', len(training_set))   
        # define input sequence
        
        timeseries_data = training_set 
        # choose a number of time steps
        n_steps = self.length
        #n_steps = 10
        X, y = prepare_data(timeseries_data, n_steps)
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        # define model
        model = Sequential()
        model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # fit model
        model.fit(X, y, epochs=30, verbose=0)
        # Saving model to disk
        #pickle.dump(model, open('model.pkl','wb'))
        # Loading model to compare the results
        
        #model = pickle.load(open('model.pkl','rb'))
        
        # demonstrate prediction for next 10 days
        sample = timeseries_data[-n_steps:]
        
        x_input = np.array(sample)
        temp_input=list(x_input)
        lst_output = []
        i=0
        while(i< n_steps):
            
            if(len(temp_input)>n_steps):
                x_input= np.array(temp_input[1:])
                print("{} day input {}".format(i,x_input))
                #print(x_input)
                x_input = x_input.reshape((1, n_steps, n_features))
                #print(x_input)
                yhat = model.predict(x_input, verbose=1)
                print("{} day output {}".format(i,yhat))
                temp_input.append(yhat[0][0])
                temp_input=temp_input[1:]
                #print(temp_input)
                lst_output.append(yhat[0][0])
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps, n_features))
                yhat = model.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.append(yhat[0][0])
                lst_output.append(yhat[0][0])
                i=i+1


        length = len(lst_output)+len(training_set)+1
        a = np.arange(1,length)
        b = np.arange(len(training_set))
        d1 =  np.array(training_set)
        d2= np.array(lst_output)
        d3 = np.concatenate([d1,d2])
        # plotting
       
        #plt.title("Line graph")
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        plt.title("LSTM ")
        plt.xlabel("No of samples")
        plt.ylabel("Price")
        plt.plot(a, d3,label='Actual Price',color ="Red", linewidth=2)
        plt.plot(b, d1,label='Predicted Price',color ="Blue", linewidth=2)
        plt.legend(loc=4)
        plt.savefig('static/lstm_result.png')
        plt.close(fig)
        return lst_output
    
    # def rnn_algo(self):
    #     dat = self.name
    #     #df_train = pd.read_csv('Stock_Price_Train.csv')
    #     dataset = pd.read_csv('stocks data/'+self.name)
    #     test_loc = len(dataset)-int(len(dataset)*0.2)
    #     train = dataset.iloc[:test_loc]
    #     test = dataset.iloc[test_loc:]
    #     training_set = train.iloc[:, 2:3].values
    #     # Feature Scaling
    #     from sklearn.preprocessing import MinMaxScaler
    #     sc = MinMaxScaler(feature_range = (0, 1))
    #     training_set_scaled = sc.fit_transform(training_set)
    #     #Creating a data structure with xx timesteps and 1 output
    #     X_train = []
    #     y_train = []
    #     for i in range(10,len(training_set)):
    #         X_train.append(training_set_scaled[i-10:i, 0])
    #         y_train.append(training_set_scaled[i, 0])
    #     X_train, y_train = np.array(X_train), np.array(y_train)
    #     # Reshaping
    #     n_features =1
    #     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_features))
    #     # Importing the Keras libraries and packages
    #     # from keras.models import Sequential
    #     # from keras.layers import Dense
    #     # from keras.layers import LSTM
    #     # from keras.layers import Dropout
    #     # Part 2 - Building the RNN
    #     # Initialising the RNN
    #     regressor = Sequential()

    #     # Adding the first LSTM layer and some Dropout regularisation
    #     regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    #     regressor.add(Dropout(0.2))

    #     # Adding a second LSTM layer and some Dropout regularisation
    #     regressor.add(LSTM(units = 50, return_sequences = True))
    #     regressor.add(Dropout(0.2))

    #     # Adding a third LSTM layer and some Dropout regularisation
    #     regressor.add(LSTM(units = 50, return_sequences = True))
    #     regressor.add(Dropout(0.2))

    #     # Adding a fourth LSTM layer and some Dropout regularisation
    #     regressor.add(LSTM(units = 50))
    #     regressor.add(Dropout(0.2))

    #     # Adding the output layer
    #     regressor.add(Dense(units = 1))

    #     # Compiling the RNN
    #     regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    #     # Fitting the RNN to the Training set
    #     regressor.fit(X_train, y_train, epochs = 10, batch_size = 32)
    #     # done model
        
    #     testing_set = dataset.iloc[test_loc:]
    #     real_stock_price = testing_set.iloc[:, 1:2].values
    #     # Getting the predicted 
    #     dataset_total = dataset['High']
    #     inputs = dataset_total[len(dataset_total) - len(testing_set) - 10:].values
    #     inputs = inputs.reshape(-1,1)
    #     inputs = sc.transform(inputs)  

    #     X_test = []
    #     for i in range(10,511):
    #         X_test.append(inputs[i-10:i, 0])

    #     X_test = np.array(X_test)
    #     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
    #     predicted_stock_price = regressor.predict(X_test)
    #     predicted_stock_price = sc.inverse_transform(predicted_stock_price) 
        
    #     new_array = np.append(real_stock_price,predicted_stock_price)
    #     length = len(new_array)
    #     x = np.arange(length)
    #     y =  predicted_stock_price
    #     z = real_stock_price
    #     # plotting
    #     fig = plt.figure(figsize=(7.2,4.8),dpi=65)
    #     plt.title("RNN_LSTM")
    #     #plt.title("Line graph")
    #     plt.xlabel("X axis")
    #     plt.ylabel("Y axis")
    #     plt.plot(y, color ="r",label='Predected')
    #     plt.plot(z, color ="g",label='actual')
    #     # plt.show()  
    #     plt.legend(loc=4)
    #     plt.savefig('static/rnn_result.png')
    #     plt.close(fig)   
    #     return predicted_stock_price[490:501]

    def cnn_algo(self):
        data = self.name
        dataset = pd.read_csv('dataset/' + self.name)
        training_set = dataset.iloc[:, 2].values

        # split a univariate sequence into samples
        def split_sequence(sequence, n_steps):
            X, y = list(), list()
            for i in range(len(sequence)):
                # find the end of this pattern
                end_ix = i + n_steps
                # check if we are beyond the sequence
                if end_ix > len(sequence) - 1:
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
                X.append(seq_x)
                y.append(seq_y)
            return np.array(X), np.array(y)

        # define input sequence
        raw_seq = training_set
        # choose a number of time step
        n_steps = 4
        # split into samples
        X, y = split_sequence(raw_seq, n_steps)
        # reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
        n_features = 1
        n_seq = 2
        n_steps = 2
        X = X.reshape((X.shape[0], n_seq, n_steps, n_features))

        # define model
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
                                  input_shape=(None, n_steps, n_features)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # fit model
        model.fit(X, y, epochs=30, verbose=1)

        a = raw_seq

        for i in range(n_steps):
            # demonstrate prediction
            x_input = array(a[-4:])
            print(x_input)
            x_input = x_input.reshape((1, n_seq, n_steps, n_features))
            yout = model.predict(x_input, verbose=1)
            print(yout)
            a = np.insert(a, 1259 + i, yout[0][0])

        length = len(a)
        x = np.arange(length)
        y = a
        z = raw_seq
        # plotting
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.title("CNN LSTM")
        # plt.title("Line graph")
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.plot(y, color="r", label='Predected')
        plt.plot(z, color="g", label='actual')
        # plt.show()
        plt.legend(loc=4)
        plt.savefig('static/cnn_result.png')
        plt.close(fig)
        return a[-11:-1]




   


