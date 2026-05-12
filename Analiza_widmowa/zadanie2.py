from scipy import signal as sig
import matplotlib.pyplot as plt
import numpy as np
# Parametry sygnału
N = 1024 # liczba próbek
fp = 500 # częstotliwość pierwszego sygnału (Hz)
fd = 1200 # częstotliwość drugiego sygnału (Hz)
fs = 8000 # częstotliwość próbkowania (Hz)
w = np.zeros(N) # szum gaussowski
s = np.zeros(N) # sygnał
n = np.arange(N) # wektor próbek
y = np.zeros(N) # sygnał z szumem
t = n / fs # wektor czasu
mi = 0
sigma = 1

w[n] = np.random.normal(mi, sigma, N)

s[n] = 0.5 * np.sin(2 * np.pi * fp * t) + np.sin(2 * np.pi * fd * t)

y[n] = s[n] + 0.1 * w[n] # sygnał z szumem


f1, Pxx_den1 = sig.periodogram(w, fs, window='boxcar', nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)
f2, Pxx_den2 = sig.periodogram(s, fs, window='boxcar', nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)
f3, Pxx_den3 = sig.periodogram(y, fs, window='boxcar', nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)

def psd_db(Pxx):
    return 10 * np.log10(np.maximum(Pxx, np.finfo(float).tiny))

plt.figure(figsize=(18, 10))
plt.suptitle('Sygnały oraz ich widmowa gęstość mocy', fontsize=14)

plt.subplot(3, 3, 1)
plt.plot(t, w, label='Szum gaussowski')
plt.title('Szum gaussowski - sygnał w czasie')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.subplot(3, 3, 2)
plt.plot(f1, psd_db(Pxx_den1), label='Okno boxcar')
plt.title('Szum gaussowski dla okna boxcar')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim(0, fs / 2)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.subplot(3, 3, 4)
plt.plot(t, s, label='Sygnał sinusoidalny')
plt.title('Sygnał sinusoidalny - sygnał w czasie')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.subplot(3, 3, 5)
plt.plot(f2, psd_db(Pxx_den2), label='Okno boxcar')
plt.title('Sygnał sinusoidalny dla okna boxcar')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim(0, fs / 2)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.subplot(3, 3, 7)
plt.plot(t, y, label='Sygnał z szumem')
plt.title('Sygnał sinusoidalny z szumem - sygnał w czasie')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.subplot(3, 3, 8)
plt.plot(f3, psd_db(Pxx_den3), label='Okno boxcar')
plt.title('Sygnał sinusoidalny z szumem dla okna boxcar')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim(0, fs / 2)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

f4, Pxx_den4 = sig.periodogram(w, fs, window='hann', nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)
f5, Pxx_den5 = sig.periodogram(s, fs, window='hann', nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)
f6, Pxx_den6 = sig.periodogram(y, fs, window='hann', nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)

plt.subplot(3, 3, 3)
plt.plot(f4, psd_db(Pxx_den4), label='Okno Hann')
plt.title('Szum gaussowski dla okna Hann')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim(0, fs / 2)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.subplot(3, 3, 6)
plt.plot(f5, psd_db(Pxx_den5), label='Okno Hann')
plt.title('Sygnał sinusoidalny dla okna Hann')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim(0, fs / 2)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.subplot(3, 3, 9)
plt.plot(f6, psd_db(Pxx_den6), label='Okno Hann')
plt.title('Sygnał sinusoidalny z szumem dla okna Hann')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim(0, fs / 2)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])

welch_params = [
    (256, 128),
    (128, 64),
    (64, 32),
]

plt.figure(figsize=(14, 10))
plt.suptitle('Widmowa gęstość mocy metodą Welcha - okno Hann, 50% nakładania', fontsize=14)

plt.subplot(3, 1, 1)
for nperseg, noverlap in welch_params:
    f_welch, Pxx_welch = sig.welch(w, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, scaling='density')
    plt.plot(f_welch, psd_db(Pxx_welch), label=f'{nperseg} próbek')
plt.title('Szum gaussowski')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim(0, fs / 2)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.subplot(3, 1, 2)
for nperseg, noverlap in welch_params:
    f_welch, Pxx_welch = sig.welch(s, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, scaling='density')
    plt.plot(f_welch, psd_db(Pxx_welch), label=f'{nperseg} próbek')
plt.title('Sygnał sinusoidalny')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim(0, fs / 2)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.subplot(3, 1, 3)
for nperseg, noverlap in welch_params:
    f_welch, Pxx_welch = sig.welch(y, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, scaling='density')
    plt.plot(f_welch, psd_db(Pxx_welch), label=f'{nperseg} próbek')
plt.title('Sygnał sinusoidalny z szumem')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim(0, fs / 2)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
