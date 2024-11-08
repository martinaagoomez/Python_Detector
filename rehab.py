import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Cargar la señal EMG desde el archivo .txt
file_path = 'C:\\Users\\teres\\Documents\\SENAL PRUEBA I\\opensignals_0c4314248023_2024-10-28_15-03-33.txt'
data = np.loadtxt(file_path, comments='#', delimiter='\t', usecols=(5,))  # Columna A1 (índice 5)

# Definir el filtro de paso de banda
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Parámetros del filtro
fs = 1000  # Frecuencia de muestreo en Hz (ajusta este valor si es diferente)
lowcut = 20.0  # Frecuencia de corte baja
highcut = 450.0  # Frecuencia de corte alta

# Filtrar la señal EMG
filtered_data = bandpass_filter(data, lowcut, highcut, fs)

# Rectificar la señal (valor absoluto)
rectified_data = np.abs(filtered_data)

# Suavizar la señal usando una media móvil
window_size = 50  # Tamaño de la ventana de suavizado
smoothed_data = np.convolve(rectified_data, np.ones(window_size) / window_size, mode='same')

# Graficar la señal EMG original, filtrada y suavizada
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(data, color='blue')
plt.title("Original EMG Signal (Channel A1)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(filtered_data, color='orange')
plt.title("Filtered EMG Signal")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(smoothed_data, color='green')
plt.title("Smoothed EMG Signal")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()