import csv
import os
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd

ecu_file_name = "../data/logs/2024-10-02 15-17-52-csv1.csv"
ecu_file_output_name = "../data/output/analysis-output/2024-10-02 15-17-52.csv"
diego_data_filename = "../data/output/ocr-output/2024-10-02 15-18-34-ocr-output.txt"
diego_time_offset = 8

combined_file_output_name = "../data/output/analysis-output/2024-10-02 15-17-52-combined.csv"

def match_ecudata_by_time(stft_data,ltft_data,pressure_data):
    stft_data.reset_index(drop=True, inplace=True)
    ltft_data.reset_index(drop=True, inplace=True)
    pressure_data.reset_index(drop=True, inplace=True)

    # List to store converted data
    grouped_data = []

    # Fit according to time
    for i in range(len(pressure_data)):
        time = pressure_data.loc[i, 'SECONDS']

        # Find STFT and LTFT values that are closest in time, that are later or equal to time from pressure data
        stft_after_time = stft_data[stft_data['SECONDS'] >= time]
        ltft_after_time = ltft_data[ltft_data['SECONDS'] >= time]

        # If there are later values, fit the one that fits the most
        if not stft_after_time.empty:
            stft_closest = stft_after_time.iloc[(stft_after_time['SECONDS'] - time).abs().argmin()]
        else:
            stft_closest = None

        if not ltft_after_time.empty:
            ltft_closest = ltft_after_time.iloc[(ltft_after_time['SECONDS'] - time).abs().argmin()]
        else:
            ltft_closest = None

        # Dodajemy dopasowany rekord tylko, gdy znaleziono dopasowane wartości
        if stft_closest is not None and ltft_closest is not None:
            grouped_data.append([
                time,  # time from pressure
                stft_closest['VALUE'],  # STFT (fit by time and later)
                ltft_closest['VALUE'],  # LTFT (fit by time and later)
                pressure_data.loc[i, 'VALUE']  # Pressure
            ])

    # Transform list to DataFrame
    return pd.DataFrame(grouped_data, columns=['Time', 'STFT', 'LTFT', 'Pressure'])

def match_ecu_with_diego(ecudata,diegodata):
    grouped_data = []
    for iterator in range(len(ecudata)):
        ecutime = ecudata.loc[iterator,'Time']
        time_diego_closest_idx = (diegodata['Time'] - ecutime).abs().argmin()
        #time_diego_closest_row = diegodata.iloc[time_diego_closest_idx]

        grouped_data.append({
            'Time': ecutime,
            'STFT': ecudata.loc[iterator,'STFT'],
            'LTFT': ecudata.loc[iterator,'LTFT'],
            'Pressure': ecudata.loc[iterator,'Pressure'],
            'Pressure_Diego': diegodata.loc[iterator,'Pressure'],
            'RPM': diegodata.loc[time_diego_closest_idx,'RPM'],
            'Gasoline': diegodata.loc[time_diego_closest_idx,'Gasoline'],
            'Gas': diegodata.loc[time_diego_closest_idx,'Gas'],
            'AirPressure': diegodata.loc[time_diego_closest_idx,'AirPressure']
        })

    return pd.DataFrame(grouped_data, columns=['Time', 'STFT', 'LTFT', 'Pressure', 'Pressure_Diego', 'RPM', 'Gasoline', 'Gas', 'AirPressure'])


def main():
    ecu_data = pd.read_csv(ecu_file_name, delimiter=';')

    ecu_data.drop('UNITS', inplace=True, axis=1)
    ecu_data.drop('Unnamed: 4', inplace=True, axis=1)

    print(ecu_data)

    ecu_data = ecu_data.loc[ecu_data['PID'] != 'Calculated boost']  # Get rid of unnecessary value
    print(ecu_data)

    # Split by groups 
    stft_data = ecu_data[ecu_data['PID'] == 'Short term fuel % trim - Bank 1']
    ltft_data = ecu_data[ecu_data['PID'] == 'Long term fuel % trim - Bank 1']
    pressure_data = ecu_data[ecu_data['PID'] == 'Intake manifold absolute pressure']

    print(len(stft_data))
    print(len(ltft_data))
    print(len(pressure_data))

    df_grouped = match_ecudata_by_time(stft_data, ltft_data, pressure_data)

    # Show transformed table
    print(df_grouped)

    df_grouped.to_csv(ecu_file_output_name, index=False)

    diego_data = pd.read_csv(diego_data_filename,sep=';')

    #Apply time offset so everything matches

    diego_data['Time'] += diego_time_offset

    result = match_ecu_with_diego(df_grouped,diego_data)

    print(result)

    # Draw graph

    result = result.loc[result['LTFT'] <= 90]

    result.to_csv(combined_file_output_name,index=False)

    # Prepare data
    result_pivot = result.pivot_table(index='RPM', columns='AirPressure', values='LTFT')

    # Create axis X, Y i Z as 3D mesh
    X, Y = np.meshgrid(result_pivot.columns.values, result_pivot.index.values)
    Z = result_pivot.values  # Z is 2d after pivot table

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis")

    plt.show()

if __name__ == "__main__":
    main()