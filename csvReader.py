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
for i in range(len(pressure_data)):
    time = pressure_data.loc[i, 'SECONDS']

    # Znajdź najbliższe czasowo wartości STFT i LTFT, które są późniejsze lub równe czasowi ciśnienia
    stft_after_time = stft_data[stft_data['SECONDS'] >= time]
    ltft_after_time = ltft_data[ltft_data['SECONDS'] >= time]

    # Jeśli są wartości późniejsze, dopasuj najbliższą w czasie
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
            time,  # czas z ciśnienia
            stft_closest['VALUE'],  # STFT (dopasowane czasowo i późniejsze)
            ltft_closest['VALUE'],  # LTFT (dopasowane czasowo i późniejsze)
            pressure_data.loc[i, 'VALUE']  # Ciśnienie
        ])

# Przekształć listę w DataFrame
df_grouped = pd.DataFrame(grouped_data, columns=['Time', 'STFT', 'LTFT', 'Pressure'])

# Wyświetl przekształconą tabelę
print(df_grouped)
