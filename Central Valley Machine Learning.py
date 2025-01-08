import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
import traces
import glob

#My plan with this code is to write a cleaner and better version of insar time series machine learning.
#I am going to just work on LSTM and see if it beats a persistence model after using exponential moving average
#to smooth out the data.
def build_cnn (trainX,trainY,testX,testY,look_back,train_scaler,test_scaler,number_of_neurons,batch_size_number,number_of_epochs):
    #Function to build the 1D CNN network and make predictions
    number_of_timesteps = np.shape(trainX) [1]
    number_of_features = np.shape(trainX) [2]
    
    model = Sequential()
    model.add(Conv1D(filters=64,kernel_size=3,activation='relu',input_shape=(number_of_timesteps,number_of_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50,activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mean_squared_error')
    
    model.fit(trainX,trainY,batch_size=batch_size_number,epochs=number_of_epochs,verbose=0)
    
    #Predict
    cnn_trainPredict = model.predict(trainX)
    cnn_testPredict = model.predict(testX)
    
    #Inverse normalize
    cnn_trainPredict = inverse_normalize(cnn_trainPredict,train_scaler)
    trainY = np.reshape(trainY, (len(trainY),1))
    trainY = inverse_normalize(trainY,train_scaler)
    cnn_testPredict = inverse_normalize(cnn_testPredict,test_scaler)
    testY = np.reshape(testY, (len(testY),1))
    testY = inverse_normalize(testY,test_scaler)
    
    #Calculate mean squared error
    cnn_train_error = mean_squared_error(cnn_trainPredict,trainY)
    cnn_test_error = mean_squared_error(cnn_testPredict,testY)
    
    print('CNN Train MSE is: %f' %(cnn_train_error))
    print('CNN Test MSE is: %f' %(cnn_test_error))
    
    
    return cnn_trainPredict, cnn_testPredict, cnn_train_error, cnn_test_error
    
    
    
    
    
    
    
def linear_fit (train,test,look_back):
    #Function to predict values using linear interpolation and calculates mse
    x=[]
    for i in range(1,look_back+1):
        x.append(i)
        
        
    linear_train_prediction=[]
    for i in range(len(train)-look_back):
        y=[]
        for j in range(look_back):
            y.append(train[i+j])
        p=np.polyfit(x,y,deg=1)
        linear_train_prediction.append(p[0]*(look_back+1)+p[1])
    
    
    
    linear_test_prediction=[]
    for i in range(len(test)-look_back):
        y=[]
        for j in range(look_back):
            y.append(test[i+j])
        p=np.polyfit(x,y,deg=1)
        linear_test_prediction.append(p[0]*(look_back+1)+p[1])
        
        
    #Calculating MSE
    train_comparison=train[look_back:len(train)]
    test_comparison=test[look_back:len(test)]
    
    linear_train_error = mean_squared_error(train_comparison, linear_train_prediction)
    linear_test_error = mean_squared_error(test_comparison, linear_test_prediction)
    
    
        
    return linear_train_prediction,linear_test_prediction,linear_train_error,linear_test_error
        




def plot_lstm_future_graphs (times_regularized, actual_displacements, look_back, train, futurePredict):
    #Function to plot the LSTM performance
    plt.plot(times_regularized,actual_displacements,label="Smoothed Displacements")
    plt.plot(times_regularized[len(train)+look_back:len(times_regularized)],futurePredict,label="LSTM Future Prediction")
    plt.title('Line of Sight Displacement of Madera, CA')
    plt.xlabel('Time')
    plt.ylabel('Displacements (mm)')
    plt.xticks(rotation=45)
    plt.legend(loc='upper right')
    plt.show()
    

def build_lstm_future (trainX,trainY,testX,testY,look_back,train_scaler,test_scaler,number_of_neurons,batch_size_number,number_of_epochs):
    #Function to build the LSTM network and make predictions
    number_of_timesteps = np.shape(trainX) [1]
    number_of_features = np.shape(trainX) [2]
    model = Sequential()
    model.add(LSTM(number_of_neurons,input_shape=(number_of_timesteps,number_of_features)))
    model.add(Dense(1))
    #metrics is just what I see (with verbose). Loss is the function that is actually optimized.
    model.compile(optimizer='RMSprop',loss='mean_squared_error')
    
    model.fit(trainX,trainY,batch_size=batch_size_number,epochs=number_of_epochs,verbose=0)
    model.summary()
    #Predict
    futurePredict=[]
    bob=np.zeros((1,look_back,1))
    print(bob)
    for i in range(look_back):
        bob[0][i][0] = testX[0][i][0]
    print(bob)
    #y_hat = model(bob)
    y_hat = model.predict(bob,batch_size=2)
    print(y_hat)
    futurePredict.append(y_hat[0][0])
    
    for i in range(len(testY)-1):
        for j in range(1,look_back):
            bob[0][j-1][0] = bob[0][j][0]
        bob[0][look_back-1][0] = y_hat[0][0]
        #y_hat = model(bob)
        y_hat = model.predict(bob,batch_size=2,verbose=0)
        futurePredict.append(y_hat[0][0])
    
    #Inverse normalize
    futurePredict = np.reshape(futurePredict, (len(futurePredict),1))
    futurePredict = inverse_normalize(futurePredict,test_scaler)
    testY = np.reshape(testY, (len(testY),1))
    testY = inverse_normalize(testY,test_scaler)
    
    #Calculate mean squared error
    future_error = mean_squared_error(futurePredict,testY)
    
    print('Future error is: %f' %(future_error))
    
    return futurePredict, future_error

def time_difference_data(displacements,times):
    #Function to time difference the data
    time_differenced_data=[]
    for i in range(0,len(displacements)-1):
        time_differenced_data.append(displacements[i+1]-displacements[i])
    
    times_differenced=times[:len(times)-1]
    
    return time_differenced_data,times_differenced

def plot_distribution(iterations,batch_size_number,number_of_neurons,number_of_epochs,trainX,trainY,testX,testY,look_back,train_scaler,test_scaler):
    #Function to plot distribution of MSE (histogram) of optimized model
    train_mse_list=[]
    test_mse_list=[]
    for i in range(iterations):
        trainPredict,testPredict,train_error,test_error = build_lstm(trainX,trainY,testX,testY,look_back,train_scaler,test_scaler,number_of_neurons,batch_size_number,number_of_epochs)
        train_mse_list.append(train_error)
        test_mse_list.append(test_error)
        
    plt.hist(train_mse_list)
    plt.title('Train MSE Distribution')
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.show()
    plt.hist(test_mse_list)
    plt.title('Test MSE Distribution')
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.show()
    
    
def plot_lstm_scatter(trainPredict,testPredict,actual_displacements,look_back,train):
    #Function to plot the scatter plot of actual vs prediction
    x=[]
    y=[]
    x.append(np.min(actual_displacements[look_back:len(train)]))
    x.append(np.max(actual_displacements[look_back:len(train)]))
    y.append(np.min(actual_displacements[look_back:len(train)]))
    y.append(np.max(actual_displacements[look_back:len(train)]))
    plt.scatter(actual_displacements[look_back:len(train)],trainPredict,color='black',marker='x')
    plt.plot(x,y,color='red')
    plt.title('Scatter Plot of Training Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()
    x=[]
    y=[]
    x.append(np.min(actual_displacements[len(train)+look_back:len(actual_displacements)]))
    x.append(np.max(actual_displacements[len(train)+look_back:len(actual_displacements)]))
    y.append(np.min(actual_displacements[len(train)+look_back:len(actual_displacements)]))
    y.append(np.max(actual_displacements[len(train)+look_back:len(actual_displacements)]))
    plt.scatter(actual_displacements[len(train)+look_back:len(actual_displacements)],testPredict,color='black',marker='x')
    plt.plot(x,y,color='red')
    plt.title('Scatter Plot of Test Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()
    
    

def plot_lstm_graphs (times_regularized, actual_displacements, look_back, train, trainPredict, testPredict):
    #Function to plot the LSTM performance
    plt.plot(times_regularized,actual_displacements,label="Smoothed Displacements")
    plt.plot(times_regularized[look_back:len(train)],trainPredict,label="LSTM Train Prediction")
    plt.plot(times_regularized[len(train)+look_back:len(times_regularized)],testPredict,label="LSTM Test Prediction")
    plt.title('Line of Sight Displacement of Madera, CA')
    plt.xlabel('Time')
    plt.ylabel('Displacements (mm)')
    plt.xticks(rotation=45)
    plt.legend(loc='upper right')
    plt.show()
    
def build_lstm (trainX,trainY,testX,testY,look_back,train_scaler,test_scaler,number_of_neurons,batch_size_number,number_of_epochs):
    #Function to build the LSTM network and make predictions
    number_of_timesteps = np.shape(trainX) [1]
    number_of_features = np.shape(trainX) [2]
    model = Sequential()
    model.add(LSTM(number_of_neurons,input_shape=(number_of_timesteps,number_of_features)))
    model.add(Dense(1))
    #metrics is just what I see (with verbose). Loss is the function that is actually optimized.
    model.compile(optimizer='RMSprop',loss='mean_squared_error')
    
    model.fit(trainX,trainY,batch_size=batch_size_number,epochs=number_of_epochs,verbose=0)
    
    #Predict
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    
    #Inverse normalize
    trainPredict = inverse_normalize(trainPredict,train_scaler)
    trainY = np.reshape(trainY, (len(trainY),1))
    trainY = inverse_normalize(trainY,train_scaler)
    testPredict = inverse_normalize(testPredict,test_scaler)
    testY = np.reshape(testY, (len(testY),1))
    testY = inverse_normalize(testY,test_scaler)
    
    #Calculate mean squared error
    train_error = mean_squared_error(trainPredict,trainY)
    test_error = mean_squared_error(testPredict,testY)
    
    print('LSTM Train MSE is: %f' %(train_error))
    print('LSTM Test MSE is: %f' %(test_error))
    
    
    return trainPredict, testPredict, train_error, test_error
    
def inverse_normalize(dataset,scaler):
    #Function to invert values back to original scaling
    inverted_dataset = scaler.inverse_transform(dataset)
    
    return inverted_dataset

def normalize_dataset(dataset):
    #Function to normalize dataset to a range of 0,1
    dataset_new = np.reshape(dataset,(len(dataset),1))
    scaler = MinMaxScaler(feature_range=(0,1))
    dataset_normalized = scaler.fit_transform(dataset_new)
    
    return dataset_normalized,scaler
    
def create_dataset(dataset,look_back):
    #Function to create trainX,testX,trainY,testY in proper format to input into LSTM
    #X has shape(number of samples, number of timesteps, number of features = 1 since only displacement)
    #Y has shape (number of samples)
    number_of_samples = len(dataset) - look_back
    number_of_timesteps = look_back
    number_of_features = 1
    X = np.zeros((number_of_samples,number_of_timesteps))
    Y = np.zeros(number_of_samples)
    counterY = 0
    for i in range(0,len(dataset) - look_back):
        Y[counterY] = dataset[i+look_back]
        counterX=0
        for j in range(0,number_of_timesteps):
            X[counterY][counterX] = dataset[i+j]
            counterX = counterX + 1
        counterY = counterY + 1
        
    X = np.reshape(X,(number_of_samples,number_of_timesteps,number_of_features))
    
    return X,Y

def baseline_performance (train, test, look_back, times_regularized, displacements):
    #Function to calculate the minimum error (MSE) to beat using a simple averaging method
    #It also plots the graphs
    actual_values_train = train[look_back:len(train)]
    actual_values_test = test[look_back:len(test)]
    
    predicted_values_train = []
    predicted_values_test = []
    
    for i in range(0,len(train)-look_back):
        temporary=[]
        for j in range(0,look_back):
            temporary.append(train[i+j])
        predicted_values_train.append(np.mean(temporary))
        
    for i in range(0,len(test)-look_back):
        temporary=[]
        for j in range(0,look_back):
            temporary.append(test[i+j])
        predicted_values_test.append(np.mean(temporary))
        
    baseline_MSE_train = mean_squared_error(actual_values_train,predicted_values_train)
    baseline_MSE_test = mean_squared_error(actual_values_test,predicted_values_test)
    
    print('The train MSE to beat is: %f' %(baseline_MSE_train))
    print('The test MSE to beat is: %f' %(baseline_MSE_test))
    
    #Plotting
    times_regularized_train = times_regularized[look_back:len(train)]
    times_regularized_test = times_regularized[len(train)+look_back:len(times_regularized)]
    
    plt.plot(times_regularized,displacements,label="Smoothed Displacements")
    plt.plot(times_regularized_train,predicted_values_train,label="Train Averaging")
    plt.plot(times_regularized_test,predicted_values_test,label="Test Averaging")
    plt.ylabel("Displacements (mm)")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.title('Line of Sight Displacement of Madera, CA')
    plt.legend(loc="upper right")
    plt.show()
    

def split_dataset (dataset,percentage_training):
    #Function that splits the dataset into train and test given a percentage.
    train_size = int(len(dataset)*percentage_training)
    train = dataset[0:train_size]
    test = dataset[train_size:len(dataset)]
    
    return train,test

def regularize_time_intervals (dataset,times,time_delta):
    #Function that regularizes the time intervals of a time series by linearly interpolating the gaps
    ts_test = traces.TimeSeries(zip(times,dataset))
    
    bob=ts_test.sample(
        sampling_period=datetime.timedelta(days=time_delta),
        start=times[0],
        end=times[len(times)-1],
        interpolate='linear')
    
    new_displacements=[]
    new_times=[]
    for i in range(0,len(bob)):
        new_displacements.append(bob[i][1])
        new_times.append(bob[i][0])
        
    return new_displacements, new_times

def average_datasets(dataset_matrix):
    #Function that takes the average of all the displacements at each time to form one averaged dataset
    #Returns a list. (Converts the numpy array to a list for easier later use)
    averaged_dataset = np.average (dataset_matrix, axis = 1)
    
    return list(averaged_dataset)

def plot_graphs(dataset1,dataset2,times):
    
    plt.plot(times,dataset1,label="Averaged Displacements")
    plt.plot(times,dataset2, label="Smoothed Displacements")
    plt.ylabel('Displacement (mm)')
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.title('Line of Sight Displacement of Madera, CA')
    plt.legend(loc="upper right")
    plt.show()

def smooth_dataset_ema (dataset,alpha):
    
    ema_displacements = []
    ema_displacements.append(dataset[0])
    for i in range (1,len(dataset)):
        ema_displacements.append(alpha*dataset[i]+(1-alpha)*ema_displacements[i-1])

    return ema_displacements

def read_lines (file_path):
    #Function that reads all the lines of a txt file. The first 8 lines of each txt file
    #are just information about the time series. The actual displacements/times start from index 8.
    with open(file_path) as file:
        lines = file.readlines()
        
    return lines

def import_datasets (files_location,starting_index):
    #Function that imports the data from txt files and returns displacements of all files (numpy matrix)
    #and associated times (list (1d since all txt files have same times))
    file_paths = glob.glob(files_location)
    number_of_files = len(file_paths)
    
    lines=[]
    for i in range(0, number_of_files):
        lines.append(read_lines(file_paths[i]))
    
    total_number_of_lines = len(lines[0])
    number_of_displacements = total_number_of_lines - starting_index
    
    displacements = np.zeros( (number_of_displacements, number_of_files) )
    times = []
    #I am assuming that the time series all have the same nan values at the same dates/rows.
    nan_row_indices_to_be_deleted=[]
    #For each file
    for k in range (0, number_of_files):
        j=0
        #Go through the useful lines
        for i in range ( starting_index , total_number_of_lines ):
            temporary = lines[k][i].split()
            if (temporary[1] != 'nan'):
                
                #No need to fill out 'times' hundreds of times. This if condition cuts back on running time.
                if (k == 0):
                    times.append(datetime.datetime.strptime(temporary[0],"%Y%m%d"))
                    
                displacements[j][k] = float(temporary[1])
            else:
                if (k==0):
                    nan_row_indices_to_be_deleted.append(j)
            j = j+1
            
    #Delete the rows of zeros in displacements that resulted from initializing the displacement matrix 
    #into zeros and not assigning any values to those rows because there is a nan in the data.
    displacements = np.delete(displacements, nan_row_indices_to_be_deleted, axis = 0)
    
    
    return (displacements,times)

if __name__ == "__main__" :
    
    #Track running time
    start=datetime.datetime.now()
    
    #Import the datasets into one dataset matrix (np.array) and the times (list)
    files_location = 'C:\\Users\\user\\Desktop\\Rundle Research\\Research Images\\Subsidence in central valley (potential paper)\\Madera time series data\\*txt'
    starting_index = 8 #Data starts at index 8
    displacements, times = import_datasets(files_location,starting_index)
    
    #Average the datasets into one
    displacements_averaged = average_datasets(displacements)
    
    #Regularize time intervals in averaged time series
    time_delta = 6 #in days
    displacements_averaged_regularized, times_regularized = regularize_time_intervals(displacements_averaged,times,time_delta)
    
    
    #Smooth the averaged and regularized time series
    alpha = 0.17
    displacements_averaged_regularized_smoothed = smooth_dataset_ema(displacements_averaged_regularized,alpha)
    
    #Plot displacements and smoothed displacements to see which alpha value to use
    #plot_graphs(displacements_averaged_regularized, displacements_averaged_regularized_smoothed, times_regularized)
    
    #Split dataset into training and testing
    training_percentage = 0.8
    train,test = split_dataset(displacements_averaged_regularized_smoothed,training_percentage)
    
    #Calculating baseline error to beat
    look_back = 4 #the number of time steps that I am using to calculate the very next timestep
    #baseline_performance(train,test,look_back,times_regularized,displacements_averaged_regularized_smoothed)
    
    
    
    #PLOTTING FOR PAPER (START)
    # actual_values_train = train[look_back:len(train)]
    # actual_values_test = test[look_back:len(test)]
    
    # predicted_values_train = []
    # predicted_values_test = []
    
    # for i in range(0,len(train)-look_back):
    #     temporary=[]
    #     for j in range(0,look_back):
    #         temporary.append(train[i+j])
    #     predicted_values_train.append(np.mean(temporary))
        
    # for i in range(0,len(test)-look_back):
    #     temporary=[]
    #     for j in range(0,look_back):
    #         temporary.append(test[i+j])
    #     predicted_values_test.append(np.mean(temporary))
        
    # baseline_MSE_train = mean_squared_error(actual_values_train,predicted_values_train)
    # baseline_MSE_test = mean_squared_error(actual_values_test,predicted_values_test)
    
    # print('The train MSE to beat is: %f' %(baseline_MSE_train))
    # print('The test MSE to beat is: %f' %(baseline_MSE_test))
    
    # times_regularized_train = times_regularized[look_back:len(train)]
    # times_regularized_test = times_regularized[len(train)+look_back:len(times_regularized)]
    
    
    # fig,ax=plt.subplots(2,1)
    # fig.set_size_inches(16,9)
    # ax[0].plot(times_regularized,displacements_averaged_regularized,color='black',label="Averaged Displacements")
    # ax[0].plot(times_regularized,displacements_averaged_regularized_smoothed,color='red', label="Smoothed Displacements")
    # ax[0].set_ylabel('Displacement (mm)')
    # ax[0].tick_params(axis='x', labelrotation = -45)
    # ax[0].legend(loc="upper right")
    # ax[0].grid()
    
    # ax[1].plot(times_regularized,displacements_averaged_regularized_smoothed,color='red',label="Smoothed Displacements")
    # ax[1].plot(times_regularized_train,predicted_values_train,color='green',label="Baseline on Train")
    # ax[1].plot(times_regularized_test,predicted_values_test,color='blue',label="Baseline on Test")
    # ax[1].set_ylabel("Displacements (mm)")
    # ax[1].tick_params(axis='x', labelrotation = -45)
    # ax[1].legend(loc="upper right")
    # ax[1].grid()
    # plt.rcParams.update({'font.size': 15})
    # #plt.savefig('baseline.pdf')
    # plt.show()
    #PLOTTING FOR PAPER (END)
    
    
    
    #Normalize datasets
    train_normalized,train_scaler = normalize_dataset(train)
    test_normalized,test_scaler = normalize_dataset(test)
    
    #Create input and output (X,Y) for the LSTM
    trainX,trainY = create_dataset(train_normalized,look_back)
    testX,testY = create_dataset(test_normalized,look_back)
    
    #Build LSTM model with the best hyperparameters choice concluded from the manual tuning
    #Ratio of 1.5 seems really good for some reason (3,2)
    number_of_neurons = 5
    batch_size_number = 90
    number_of_epochs = 5000
    # with tf.device('/cpu:0'):
    #     trainPredict, testPredict, train_error, test_error = build_lstm(trainX,trainY,testX,testY,look_back,train_scaler,test_scaler,number_of_neurons,batch_size_number,number_of_epochs)
    
    #Plot LSTM results
    #plot_lstm_graphs(times_regularized,displacements_averaged_regularized_smoothed,look_back,train,trainPredict,testPredict)
    
    #Plot LSTM actual future predictions
    #futurePredict, future_error = build_lstm_future(trainX, trainY, testX, testY, look_back, train_scaler, test_scaler, number_of_neurons, batch_size_number, number_of_epochs)
    #plot_lstm_future_graphs(times_regularized,displacements_averaged_regularized_smoothed,look_back,train,futurePredict)
    
    #PLOTTING FOR PAPER (START)
    # trainY_copy=trainY
    # testY_copy=testY
    # number_of_timesteps = np.shape(trainX) [1]
    # number_of_features = np.shape(trainX) [2]
    # model = Sequential()
    # model.add(LSTM(number_of_neurons,input_shape=(number_of_timesteps,number_of_features)))
    # model.add(Dense(1))
    # #metrics is just what I see (with verbose). Loss is the function that is actually optimized.
    # model.compile(optimizer='RMSprop',loss='mean_squared_error')
    
    # model.fit(trainX,trainY,batch_size=batch_size_number,epochs=number_of_epochs,verbose=0)
    
    # #Predict
    # trainPredict = model.predict(trainX)
    # testPredict = model.predict(testX)
    
    # #Inverse normalize
    # trainPredict = inverse_normalize(trainPredict,train_scaler)
    # trainY = np.reshape(trainY, (len(trainY),1))
    # trainY = inverse_normalize(trainY,train_scaler)
    # testPredict = inverse_normalize(testPredict,test_scaler)
    # testY = np.reshape(testY, (len(testY),1))
    # testY = inverse_normalize(testY,test_scaler)
    
    # #Calculate mean squared error
    # train_error = mean_squared_error(trainPredict,trainY)
    # test_error = mean_squared_error(testPredict,testY)
    
    # print('LSTM Train MSE is: %f' %(train_error))
    # print('LSTM Test MSE is: %f' %(test_error))
    
    
    
    # trainY=trainY_copy
    # testY=testY_copy
    
    # number_of_timesteps = np.shape(trainX) [1]
    # number_of_features = np.shape(trainX) [2]
    # model = Sequential()
    # model.add(LSTM(number_of_neurons,input_shape=(number_of_timesteps,number_of_features)))
    # model.add(Dense(1))
    # #metrics is just what I see (with verbose). Loss is the function that is actually optimized.
    # model.compile(optimizer='RMSprop',loss='mean_squared_error')
    
    # model.fit(trainX,trainY,batch_size=batch_size_number,epochs=number_of_epochs,verbose=0)
    # #Predict
    # futurePredict=[]
    # bob=np.zeros((1,look_back,1))
    # for i in range(look_back):
    #     bob[0][i][0] = testX[0][i][0]
    # #y_hat = model(bob)
    # y_hat = model.predict(bob,batch_size=2)
    # futurePredict.append(y_hat[0][0])
    
    # for i in range(len(testY)-1):
    #     for j in range(1,look_back):
    #         bob[0][j-1][0] = bob[0][j][0]
    #     bob[0][look_back-1][0] = y_hat[0][0]
    #     #y_hat = model(bob)
    #     y_hat = model.predict(bob,batch_size=2,verbose=0)
    #     futurePredict.append(y_hat[0][0])
    
    # #Inverse normalize
    # futurePredict = np.reshape(futurePredict, (len(futurePredict),1))
    # futurePredict = inverse_normalize(futurePredict,test_scaler)
    # testY = np.reshape(testY, (len(testY),1))
    # testY = inverse_normalize(testY,test_scaler)
    
    # #Calculate mean squared error
    # future_error = mean_squared_error(futurePredict,testY)
    
    # print('Future error is: %f' %(future_error))
    
    
    # fig,ax=plt.subplots(2,1)
    # fig.set_size_inches(16,9)
    # ax[0].plot(times_regularized,displacements_averaged_regularized_smoothed,color='red',label="Smoothed Displacements")
    # ax[0].plot(times_regularized[look_back:len(train)],trainPredict,color='green',label="LSTM Train Prediction")
    # ax[0].plot(times_regularized[len(train)+look_back:len(times_regularized)],testPredict,color='blue',label="LSTM Test Prediction")
    # ax[0].set_ylabel('Displacements (mm)')
    # ax[0].tick_params(axis='x', labelrotation = -45)
    # ax[0].legend(loc='upper right')
    # ax[0].grid()
    
    # ax[1].plot(times_regularized,displacements_averaged_regularized_smoothed,color='red',label="Smoothed Displacements")
    # ax[1].plot(times_regularized[len(train)+look_back:len(times_regularized)],futurePredict,color='blue',label="LSTM Future Prediction")
    # ax[1].set_ylabel('Displacements (mm)')
    # ax[1].tick_params(axis='x', labelrotation = -45)
    # ax[1].legend(loc='upper right')
    # ax[1].grid()
    # plt.rcParams.update({'font.size': 15})
    # plt.savefig('performance_prediction.pdf')
    # plt.show()
    #PLOTTING FOR PAPER (END)
    
    #Plot LSTM scatter plot
    #plot_lstm_scatter(trainPredict,testPredict,displacements_averaged_regularized_smoothed,look_back,train)
    
    #Plot distribution of MSE for optimized model
    #iterations = 0
    #number_of_neurons = 5
   # batch_size_number = 2
    #number_of_epochs = 1000
    #plot_distribution(iterations,batch_size_number,number_of_neurons,number_of_epochs,trainX,trainY,testX,testY,look_back,train_scaler,test_scaler)
    
    
    #Building 1D CNN for referee
    cnn_trainPredict,cnn_testPredict,cnn_train_error,cnn_test_error = build_cnn(trainX,trainY,testX,testY,look_back,train_scaler,test_scaler,number_of_neurons,batch_size_number,number_of_epochs)
    
    #PLOTTING FOR PAPER (START)
    # fig,ax=plt.subplots()
    # fig.set_size_inches(16,4.5)
    # ax.plot(times_regularized,displacements_averaged_regularized_smoothed,color='red',label="Smoothed Displacements")
    # ax.plot(times_regularized[look_back:len(train)],cnn_trainPredict,color='green',label="CNN Train Prediction")
    # ax.plot(times_regularized[len(train)+look_back:len(times_regularized)],cnn_testPredict,color='blue',label="CNN Test Prediction")
    # ax.set_ylabel('Displacements (mm)')
    # ax.tick_params(axis='x', labelrotation = -45)
    # ax.legend(loc='upper right')
    # ax.grid()
    # plt.rcParams.update({'font.size': 15})
    # plt.savefig('cnn_plot.pdf')
    # plt.show()
    #PLOTTING FOR PAPER (END)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #TIME DIFFERENCED SECTION
    
    activate = 0
    if activate == 1:
        #I'm going to check the time-differenced data and run the same thing on it
        time_differenced_data, times_differenced = time_difference_data(displacements_averaged_regularized_smoothed,times_regularized)
        
        #Split dataset into training and testing
        training_percentage = 0.8
        train,test = split_dataset(time_differenced_data,training_percentage)
        
        #Calculating baseline error to beat
        look_back = 4 #the number of time steps that I am using to calculate the very next timestep
        #baseline_performance(train,test,look_back,times_differenced,time_differenced_data)
        
        #Normalize datasets
        train_normalized,train_scaler = normalize_dataset(train)
        test_normalized,test_scaler = normalize_dataset(test)
        
        #Create input and output (X,Y) for the LSTM
        trainX,trainY = create_dataset(train_normalized,look_back)
        testX,testY = create_dataset(test_normalized,look_back)
        
        #Build LSTM model with the same hyperparameters
        number_of_neurons = 5
        batch_size_number = 90
        number_of_epochs = 5000
        trainPredict, testPredict, train_error, test_error = build_lstm(trainX,trainY,testX,testY,look_back,train_scaler,test_scaler,number_of_neurons,batch_size_number,number_of_epochs)
        
        #Plot LSTM results
        #plot_lstm_graphs(times_differenced,time_differenced_data,look_back,train,trainPredict,testPredict)
        
        #Plot LSTM scatter plot
        #plot_lstm_scatter(trainPredict,testPredict,time_differenced_data,look_back,train)
        
        
        #PLOTTING FOR PAPER(START)
        # x=[]
        # y=[]
        # x.append(np.min(time_differenced_data[look_back:len(train)]))
        # x.append(np.max(time_differenced_data[look_back:len(train)]))
        # y=x
        # fig,ax=plt.subplots(1,2)
        # fig.set_size_inches(15,6)
        # ax[0].scatter(time_differenced_data[look_back:len(train)],trainPredict,color='black',marker='x')
        # ax[0].plot(x,y,color='red',label='y=x')
        # ax[0].set_xlabel('Actual Values')
        # ax[0].set_ylabel('Predicted Values')
        # ax[0].legend(loc='upper left')
        # x=[]
        # y=[]
        # x.append(np.min(time_differenced_data[len(train)+look_back:len(time_differenced_data)]))
        # x.append(np.max(time_differenced_data[len(train)+look_back:len(time_differenced_data)]))
        # y=x
        # ax[1].scatter(time_differenced_data[len(train)+look_back:len(time_differenced_data)],testPredict,color='black',marker='x')
        # ax[1].plot(x,y,color='red',label='y=x')
        # ax[1].set_xlabel('Actual Values')
        # ax[1].set_ylabel('Predicted Values')
        # ax[1].legend(loc='upper left')
        # plt.rcParams.update({'font.size': 15})
        # plt.savefig('scatter.pdf')
        # plt.show()
        #PLOTTING FOR PAPER (END)
    
    
    
    
    
    
    
    
    #Track running time
    print(datetime.datetime.now()-start)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    