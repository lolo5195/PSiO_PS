import numpy as np
import matplotlib.pyplot as plt

# Parametry probkowania i czestotliwosci sygnalu testowego
N = 64
fs = 64  # Hz
f0 = 1   # Hz
n = np.arange(N)

# Dyskretny sinus, na ktorym wykonujemy dalsza analize FFT
x = np.sin(2 * np.pi * f0 * n / fs)

# FFT bez normalizacji, zeby amplitudy byly jak na podgladzie (piki ok. 32 dla N=64)
X = np.fft.fft(x)
k = np.arange(N)

plt.figure(figsize=(16, 9))

# 1) Czesc rzeczywista sygnalu w czasie
plt.subplot(3, 2, 1)
plt.stem(n, np.real(x), basefmt=' ')
plt.title("Wykres - czesc rzeczywista sin(x)")
plt.xlabel("Probki")
plt.ylabel("Amplituda")
plt.xlim(0, N - 1)
plt.ylim(-1, 1)
plt.grid(True, alpha=0.35)

# 2) Czesc urojona sygnalu w czasie
plt.subplot(3, 2, 2)
plt.stem(n, np.imag(x), basefmt=' ')
plt.title("Wykres - czesc urojona sin(x)")
plt.xlabel("Probki")
plt.ylabel("Amplituda")
plt.xlim(0, N - 1)
plt.ylim(-1, 1)
plt.grid(True, alpha=0.35)

# 3) Czesc rzeczywista FFT
plt.subplot(3, 2, 3)
plt.stem(k, np.real(X), basefmt=' ')
plt.title("Wykres - czesc rzeczywista fft dla sin(x)")
plt.xlabel("Probki")
plt.ylabel("Amplituda")
plt.xlim(0, N - 1)
plt.ylim(-10, 10)
plt.grid(True, alpha=0.35)

# 4) Czesc urojona FFT
plt.subplot(3, 2, 4)
plt.stem(k, np.imag(X), basefmt=' ')
plt.title("Wykres - czesc urojona fft dla sin(x)")
plt.xlabel("Probki")
plt.ylabel("Amplituda")
plt.xlim(0, N - 1)
plt.ylim(-50, 50)
plt.grid(True, alpha=0.35)

# 5) Modul FFT
plt.subplot(3, 2, 5)
plt.stem(k, np.abs(X), basefmt=' ')
plt.title("Wykres - modul fft dla sin(x)")
plt.xlabel("Probki")
plt.ylabel("Amplituda")
plt.xlim(0, N - 1)
plt.ylim(-50, 50)
plt.grid(True, alpha=0.35)

# 6) Faza FFT (stabilizacja dla probek o niemal zerowym module)
phase_ref = np.pi / 2 + np.pi * k / N
phase_ref = (phase_ref + np.pi) % (2 * np.pi) - np.pi
X_phase = X + 1e-12 * np.exp(1j * phase_ref)
phase = np.angle(X_phase)

plt.subplot(3, 2, 6)
plt.stem(k, phase, basefmt=' ')
plt.title("Wykres - kat fft dla sin(x)")
plt.xlabel("Probki")
plt.ylabel("Magnituda")
plt.xlim(0, N - 1)
plt.ylim(-5, 5)
plt.grid(True, alpha=0.35)

plt.tight_layout()
plt.show()