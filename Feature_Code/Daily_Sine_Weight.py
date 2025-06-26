import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load data
df = pd.read_csv("~/Downloads/IBS_Wearable_Circadian/Wearable_Data/AJ_116.csv")
df['time'] = pd.to_datetime(df['time'])
df['date'] = df['time'].dt.date

# Custom sine activation
def sine_activation(x):
    return K.sin(2 * np.pi * x)

# Build model
def build_model(n=10):
    model = Sequential([
        Input(shape=(1,)),
        Dense(n, activation=sine_activation),
        Dense(1)
    ])
    return model

# Reconstruct signal
def reconstruct(t, freq, phase, amp, bias):
    t = np.array(t).flatten()
    y = sum([amp[i] * np.sin(2 * np.pi * (freq[i] * t + phase[i])) for i in range(len(freq))])
    return y + bias[0]

# Early stopping
early_stop = EarlyStopping(monitor='loss', patience=300, restore_best_weights=True)

# Fit model for each day
daily_results = []
period_amp_pairs = []

for date in df['date'].unique():
    day_df = df[df['date'] == date]
    if len(day_df) < 50:
        continue

    t = (day_df['time'] - pd.Timestamp(date)).dt.total_seconds().values / (60 * 60 * 24)
    y = day_df['act'].values
    t, y = t.reshape(-1, 1), y.reshape(-1, 1)

    model = build_model(n=30)
    model.compile(optimizer=Adam(0.01), loss='mae')
    model.fit(t, y, epochs=2500, batch_size=32, verbose=0, callbacks=[early_stop])

    freq = model.layers[0].get_weights()[0][0, :]
    phase = model.layers[0].get_weights()[1]
    amp = model.layers[1].get_weights()[0][:, 0]
    bias = model.layers[1].get_weights()[1]

    daily_results.append({
        'date': date, 't': t.flatten(), 'y': y.flatten(),
        'freq': freq, 'phase': phase, 'amp': amp, 'bias': bias
    })

    for f, a in zip(freq, amp):
        if abs(f) > 1e-3:
            period = 24 / abs(f)
            amplitude = abs(a)
            period_amp_pairs.append((period, amplitude))

# Extract circadian periods
circadian_summary = []

for res in daily_results:
    freq, amp = res['freq'], res['amp']
    mask = np.abs(freq) > 1e-3
    periods = 24 / np.abs(freq[mask])
    amplitudes = np.abs(amp[mask])

    idx = np.argsort(-amplitudes)
    periods, amplitudes = periods[idx], amplitudes[idx]

    circadian_mask = (periods >= 20) & (periods <= 28)
    circadian_periods = periods[circadian_mask]
    circadian_amps = amplitudes[circadian_mask]

    mean_period = np.mean(circadian_periods) if len(circadian_periods) > 0 else np.nan

    circadian_summary.append({
        'date': res['date'],
        'mean_period': mean_period,
        'periods': circadian_periods,
        'amplitudes': circadian_amps
    })

# Overall stats
mean_periods = [x['mean_period'] for x in circadian_summary if not np.isnan(x['mean_period'])]
overall_mean = np.mean(mean_periods)

# Print results
for entry in circadian_summary:
    print(f"Date: {entry['date']} - Mean Period: {entry['mean_period']:.3f} h")

print(f"\nOverall Mean Circadian Period: {overall_mean:.3f} h")

# Plot actual vs reconstructed (first 7 days)
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()

for i in range(min(7, len(daily_results))):
    res = daily_results[i]
    t, y = res['t'], res['y']
    y_hat = reconstruct(t, res['freq'], res['phase'], res['amp'], res['bias'])

    ax = axes[i]
    ax.plot(t, y, label='Actual', alpha=0.6)
    ax.plot(t, y_hat, label='Reconstructed', linestyle='--')
    ax.set_title(str(res['date']))
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Activity")
    ax.grid(True)
    ax.legend()

# Hide empty plots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.suptitle("Actual vs Reconstructed Activity (First 7 Days)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
