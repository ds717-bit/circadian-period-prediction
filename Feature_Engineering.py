import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from datetime import time

# Compute the slope (coefficient) of a linear regression line fit to (x, y) data
def get_slope(x, y):
    model = LinearRegression().fit(np.array(x).reshape(-1, 1), y)
    return model.coef_[0]

# Load CSV file and prepare datetime index for time-series analysis
def get_file(file_path):
    df = pd.read_csv(file_path)
    df["time"] = pd.to_datetime(df["time"])  # Ensure "time" column is in datetime format
    df.set_index("time", inplace=True)       # Use "time" as DataFrame index
    return df

# --------- L5, M10, and Their Timing Slopes ---------

def l5_m10_feature(file_path):
    df = get_file(file_path)

    # Resample into daily groups
    daily = df.resample("1D")
    l5_list, m10_list = [], []

    for _, group in daily:
        # Compute 5-hour (300 min) and 10-hour (600 min) rolling means
        rolling_l5 = group["act"].rolling(window=300, min_periods=1).mean()
        rolling_m10 = group["act"].rolling(window=600, min_periods=1).mean()
        l5_list.append(rolling_l5.min())  # Lowest activity (L5)
        m10_list.append(rolling_m10.max())  # Highest activity (M10)
    
    # Average across all days
    L5 = np.mean(l5_list)
    M10 = np.mean(m10_list)
    
    # To compute drift over days, extract start times of L5/M10 windows
    l5_centers, m10_centers = [], []
    days_l5, days_m10 = [], []

    for i, (day, group) in enumerate(daily):
        rolling_l5 = group["act"].rolling(window=300, min_periods=1).mean()
        rolling_m10 = group["act"].rolling(window=600, min_periods=1).mean()

        l5_time = rolling_l5.idxmin()  # Timestamp where L5 occurs
        m10_time = rolling_m10.idxmax()  # Timestamp where M10 occurs
        
        if pd.notnull(l5_time):
            l5_minutes = l5_time.hour * 60 + l5_time.minute
            l5_centers.append(l5_minutes)
            days_l5.append(i)

        if pd.notnull(m10_time):
            m10_minutes = m10_time.hour * 60 + m10_time.minute
            m10_centers.append(m10_minutes)
            days_m10.append(i)

    # Compute slope of L5 and M10 timing across days
    l5_slope = get_slope(days_l5, l5_centers)
    m10_slope = get_slope(days_m10, m10_centers)

    return L5, M10, l5_slope, m10_slope

# --------- Onset, Offset Slopes and Activity Range ---------

def onset_offset_slopes_and_range(file_path):
    df = get_file(file_path)
    daily = df.resample("1D")

    onset_time_mins, offset_time_mins = [], []
    activity_range = []
    days = []

    for i, (day, group) in enumerate(daily):
        if group.empty:
            continue

        # Smooth activity using 30-min rolling average
        arranged = group['act'].rolling(window=30, min_periods=1).mean()
        derivative = arranged.diff()  # First derivative to detect changes in activity
        
        # Look at changes in activity during morning and evening
        morning = derivative.between_time("4:00", "12:00")
        evening_part1 = derivative.between_time("18:00", "23:59")

        # Attempt to include post-midnight part of evening
        try:
            next_day = daily.get_group(day + timedelta(days=1))
            next_day_arranged = next_day['act'].rolling(window=30, min_periods=1).mean()
            next_day_derivative = next_day_arranged.diff()
            evening_part2 = next_day_derivative.between_time("00:00", "02:00")
            evening = pd.concat([evening_part1, evening_part2])
        except KeyError:
            evening = evening_part1

        # Clean NaN values
        morning_clean = morning.dropna()
        evening_clean = evening.dropna()

        # Define onset time as first sharp increase in morning activity (85th percentile)
        if not morning_clean.empty:
            threshold_onset = np.percentile(morning_clean, 85)
            onset_time = morning[morning > threshold_onset].first_valid_index()
        else:
            onset_time = None

        # Define offset time as last sharp decrease in evening activity (15th percentile)
        if not evening_clean.empty:
            threshold_offset = np.percentile(evening_clean, 15)
            offset_time = evening[evening < threshold_offset].last_valid_index()
        else:
            offset_time = None

        # Convert onset/offset timestamps to minutes since midnight
        if onset_time is not None:
            onset_time_minutes = onset_time.hour * 60 + onset_time.minute
        
        else:
            onset_time_minutes = None

        if offset_time is not None:
            # Post-midnight values are treated as part of previous day's night
            if time(0, 0) <= offset_time.time() < time(2, 0):
                offset_time_minutes = 1440 + offset_time.hour * 60 + offset_time.minute
            else:
                offset_time_minutes = offset_time.hour * 60 + offset_time.minute
        else:
            offset_time_minutes = None
            
        # Duration of high activity period
        daily_range = offset_time_minutes - onset_time_minutes if None not in (onset_time_minutes, offset_time_minutes) else None
        activity_range.append(daily_range)
        onset_time_mins.append(onset_time_minutes)
        offset_time_mins.append(offset_time_minutes)
        days.append(i)

    valid_ranges = [r for r in activity_range if r is not None]
    mean_act_range = np.mean(valid_ranges) if valid_ranges else None
    
    valid_onsets = [(d, o) for d, o in zip(days, onset_time_mins) if o is not None]
    valid_offsets = [(d, o) for d, o in zip(days, offset_time_mins) if o is not None]

    onset_slope = get_slope(*zip(*valid_onsets)) if valid_onsets else None
    offset_slope = get_slope(*zip(*valid_offsets)) if valid_offsets else None

    return onset_slope, offset_slope, mean_act_range

# --------- IV (Intra-daily Variability) and IS (Inter-daily Stability) ---------

def IV_IS_features(file_path):
    df = get_file(file_path)

    # Intra-daily Variability (IV)
    # Measures how fragmented or variable activity is within a single day
    diff_series = df["act"].diff().dropna()
    IV = np.var(diff_series) / np.var(df["act"]) if np.var(df["act"]) != 0 else 0

    # Inter-daily Stability (IS)
    # Measures how consistent the daily activity pattern is across days
    hourly = df["act"].resample("1h").mean()  
    mean_hourly_pattern = hourly.groupby(hourly.index.hour).mean() 
    total_mean = df["act"].mean()

    numerator = np.sum((mean_hourly_pattern - total_mean) ** 2)
    denominator = np.sum((hourly - total_mean) ** 2)

    IS = (24 * numerator / denominator) if denominator != 0 else 0

    return IV, IS


def main():
    sample = [
        'AJ_076', 'AJ_094', 'AJ_097', 'AJ_103', 'AJ_104', 'AJ_105', 'AJ_107', 'AJ_109', 'AJ_110',
        'AJ_113', 'AJ_115', 'AJ_116', 'AJ_119', 'AJ_120', 'AJ_121', 'AJ_122', 'AJ_123', 'AJ_124',
        'AJ_125', 'AJ_129', 'AJ_131', 'AJ_132', 'AJ_133', 'AJ_134', 'AJ_135', 'AJ_137',
        'AJ_138', 'AJ_140', 'AJ_141', 'AJ_142', 'AJ_147', 'AJ_148', 'AJ_149', 'AJ_150', 'AJ_151',
        'AJ_152', 'AJ_153', 'AJ_154', 'AJ_155', 'AJ_156', 'AJ_157', 'AJ_158', 'AJ_159', 'AJ_160',
        'AJ_161', 'AJ_162', 'AJ_163', 'AJ_164', 'AJ_165', 'AJ_166', 'AJ_167', 'AJ_168', 'AJ_169',
        'AJ_170', 'AJ_172', 'AJ_173', 'AJ_174', 'AJ_175', 'AJ_176', 'AJ_179', 'AJ_181', 'AJ_183',
        'AJ_184', 'AJ_185', 'AJ_186', 'AJ_187', 'AJ_188', 'AJ_189', 'AJ_190', 'AJ_191', 'AJ_192',
        'AJ_193', 'AJ_196', 'AJ_199', 'AJ_200', 'AJ_202', 'AJ_203', 'AJ_205', 'AJ_206', 'AJ_208',
        'AJ_213', 'AJ_214', 'AJ_217', 'AJ_220', 'AJ_223', 'AJ_226', 'AJ_234', 'AJ_235']
    data = []
    for name in sample:
        file_name = f'Downloads/IBS_Wearable_Circadian/data 6/{name}.csv'
        try:
            S_L5, S_M10, S_l5_slope, S_m10_slope = l5_m10_feature(file_name)
            S_onset_slope, S_offset_slope, S_mean_act_range = onset_offset_slopes_and_range(file_name)
            S_IV, S_IS = IV_IS_features(file_name)
            data.append([name, S_L5, S_M10, S_l5_slope, S_m10_slope,
                          S_onset_slope, S_offset_slope, S_mean_act_range, S_IV, S_IS])
        except Exception as e:
            print(f"Error processing {name}: {e}")

    feature_table = pd.DataFrame(data, columns=['name', 'L5', 'M10', 'L5 Slope', 'M10 Slope','Onset Slope', 'Offset Slope', 'Activity Range', 'IV', 'IS'])
    print(feature_table)
    
    feature_table.to_csv("Downloads/circadian_features.csv", index=False)

if __name__ == "__main__":
    main()


