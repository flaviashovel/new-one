import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import LSTM, TimeDistributed, RepeatVector, Dense, Flatten
from keras.optimizers import Adam

n_steps = 1
subseq = 1
no2 = pd.read_csv("no2.csv", parse_dates=['pubtime'], index_col='pubtime')
no2_value = no2.values

def train_test_split(df, test_len=48):
    """
    Split data in training and testing. Use 48 hours as testing.
    """
    train, test = df[:-test_len], df[-test_len:]
    return train, test


def split_data(sequences, n_steps):
    """
    Preprocess data returning two arrays.
    """
    x, y = [], []
    for i in range(len(sequences)):
        end_x = i + n_steps

        if end_x > len(sequences):
            break
        x.append(sequences[i:end_x, :-1])
        y.append(sequences[end_x - 1, -1])

    return np.array(x), np.array(y)


def CNN_LSTM(x, y, x_val, y_val):
    """
    CNN-LSTM model.
    """
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=14, kernel_size=1, activation="sigmoid",
                                     input_shape=(None, x.shape[2], x.shape[3]))))
    model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(21, activation="tanh", return_sequences=True))
    model.add(LSTM(14, activation="tanh", return_sequences=True))
    model.add(LSTM(7, activation="tanh"))
    model.add(Dense(3, activation="sigmoid"))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=['mse'])
    history = model.fit(x, y, epochs=250, batch_size=36,
                        verbose=0, validation_data=(x_val, y_val))

    return model, history


# split and resahpe data
train, test = train_test_split(no2)

train_x = train.drop(columns="DC_POWER", axis=1).to_numpy()
train_y = train["DC_POWER"].to_numpy().reshape(len(train), 1)

test_x = test.drop(columns="DC_POWER", axis=1).to_numpy()
test_y = test["DC_POWER"].to_numpy().reshape(len(test), 1)

# scale data
scaler_x = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))

train_x = scaler_x.fit_transform(train_x)
train_y = scaler_y.fit_transform(train_y)

test_x = scaler_x.transform(test_x)
test_y = scaler_y.transform(test_y)

# shape data into CNN-LSTM format [samples, subsequences, timesteps, features] ORIGINAL
train_data_np = np.hstack((train_x, train_y))
x, y = split_data(train_data_np, n_steps)
x_subseq = x.reshape(x.shape[0], subseq, x.shape[1], x.shape[2])

# create validation set
x_val, y_val = x_subseq[-24:], y[-24:]
x_train, y_train = x_subseq[:-24], y[:-24]

n_features = x.shape[2]
actual = scaler_y.inverse_transform(test_y)

# run CNN-LSTM model
if __name__ == '__main__':
    #start_time = time()

    model, history = CNN_LSTM(x_train, y_train, x_val, y_val)
    prediction = []

    for i in range(len(test_x)):
        test_input = test_x[i].reshape(1, subseq, n_steps, n_features)
        yhat = model.predict(test_input, verbose=0)
        yhat_IT = scaler_y.inverse_transform(yhat)
        prediction.append(yhat_IT[0][0])

    #time_len = time() - start_time
    mse = mean_squared_error(actual.flatten(), prediction)

    #print(f'CNN-LSTM runtime: {round(time_len / 60, 2)} mins')
    print(f"CNN-LSTM MSE: {round(mse, 2)}")