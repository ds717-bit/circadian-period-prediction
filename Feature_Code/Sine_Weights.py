import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


file_name = "~/Downloads/IBS_Wearable_Circadian/Wearable_Data/AJ_116.csv" 
df = pd.read_csv(file_name)
df['time'] = pd.to_datetime(df['time'])

t0 = df['time'].min()
t = (df['time'] - t0).dt.total_seconds().values / (60 * 60 * 24 * 7)

y = df['act'].values


t = t.reshape(-1, 1)
y = y.reshape(-1, 1)

def sine_activation(x):
    return K.sin(2*np.pi*x)

def model_get(nh=10):
    model = Sequential()
    model.add(Input(shape={1,}))
    model.add(Dense(nh, activation=sine_activation))
    model.add(Dense(1, activation='linear'))
    return model

def reconstruct(t_input, freq, phase, amp, bias):
    t_input = np.array(t_input).flatten()
    output=[]
    for ti in t_input:
        val = sum([amp[i] * np.sin(2 * np.pi * (freq[i] * ti + phase[i])) for i in range(len(freq))])
        output.append(val + bias[0])
    return np.array(output)

model_path = '~/Downloads/IBS_Wearable_Circadian/Feature_Code/sine_weights.weights.h5'
cb_checkpoint = ModelCheckpoint(model_path, monitor='loss', save_best_only=True, save_weights_only=True,verbose = 1)
cb_early_stopping = EarlyStopping(monitor='loss', patience=500, restore_best_weights=True)

model = model_get(nh=10)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')

history = model.fit(t, y, epochs=10000, batch_size=32, callbacks=[cb_checkpoint, cb_early_stopping], verbose=2)

model.load_weights(model_path)
freq = model.layers[0].get_weights()[0][0,:]
phase = model.layers[0].get_weights()[1]
amp = model.layers[1].get_weights()[0][:,0]
bias = model.layers[1].get_weights()[1]
reconstructed = reconstruct(t, freq, phase, amp, bias)

plt.figure(figsize=(12, 4))
plt.plot(t, y, label="Original", alpha=0.5)
plt.plot(t, reconstructed, label="Reconstructed", linestyle="--")
plt.legend()
plt.title("INR Fit using Sine Activation")
plt.xlabel("Normalized time (0 ~ 1)")
plt.ylabel("Activity")
plt.grid(True)
plt.show()

print(freq)

