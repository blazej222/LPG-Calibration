import csv
import pandas as pd

filename = "data/logs/2024-10-02 15-17-52-csv1.csv"

data = pd.read_csv(filename,delimiter=';')

data.drop('UNITS', inplace=True, axis=1)
data.drop('Unnamed: 4', inplace=True, axis=1)

print(data)

data = data.loc[data['PID'] != 'Calculated boost'] #Get rid of unnecessary value
print(data)

# # Pivot table, aby stworzyć kolumny dla STFT, LTFT i ciśnienia
# df_pivot = data.pivot_table(index='SECONDS', columns='PID', values='VALUE', aggfunc='first')
#
# # Resetuj indeks, aby czas był zwykłą kolumną
# df_pivot.reset_index(inplace=True)
#
# print(df_pivot)

# Podziel dane na grupy według parametrów
stft_data = data[data['PID'] == 'Short term fuel % trim - Bank 1']
ltft_data = data[data['PID'] == 'Long term fuel % trim - Bank 1']
pressure_data = data[data['PID'] == 'Intake manifold absolute pressure']

# Zresetuj indeksy, aby ułatwić operacje
stft_data.reset_index(drop=True, inplace=True)
ltft_data.reset_index(drop=True, inplace=True)
pressure_data.reset_index(drop=True, inplace=True)

# Lista do przechowywania przekształconych danych
grouped_data = []

# Dopasowanie na podstawie czasu
for i in range(len(stft_data)):
    time = stft_data.loc[i, 'SECONDS']

    # Znajdź najbliższe czasowo wartości LTFT i ciśnienia
    ltft_closest = ltft_data.iloc[(ltft_data['SECONDS'] - time).abs().argmin()]
    pressure_closest = pressure_data.iloc[(pressure_data['SECONDS'] - time).abs().argmin()]

    # Dodajemy dopasowany rekord
    grouped_data.append([
        time,  # czas z STFT
        stft_data.loc[i, 'VALUE'],  # STFT
        ltft_closest['VALUE'],  # LTFT (dopasowane czasowo)
        pressure_closest['VALUE']  # Ciśnienie (dopasowane czasowo)
    ])

# Przekształć listę w DataFrame
df_grouped = pd.DataFrame(grouped_data, columns=['Time', 'STFT', 'LTFT', 'Pressure'])

# Wyświetl przekształconą tabelę
print(df_grouped)
