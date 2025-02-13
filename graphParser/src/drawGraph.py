from matplotlib import pyplot as plt
import pandas as pd

ecu_file = "../data/output/analysis-output/2024-10-02 15-17-52.csv"
diego_file = "../data/output/ocr-output/2024-10-02 15-18-34-ocr-output.txt"

ecu = pd.read_csv(ecu_file)
diego = pd.read_csv(diego_file,sep=';')

diego['Time'] += 8

print(diego)

fig,ax = plt.subplots(2)

ax[0].plot(ecu['Time'],ecu['Pressure'])
ax[0].set_title("ECU Pressure vs Time")

ax[1].plot(diego['Time'],diego['Pressure'])
ax[1].set_title("Diego Pressure vs Time")

plt.show()
