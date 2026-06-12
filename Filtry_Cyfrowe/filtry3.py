import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# Parametry z zadania
fs = 8000          # czestotliwosc probkowania [Hz]
fg = 500           # czestotliwosc graniczna [Hz]
N = 3              # rzad filtrow

# Parametry dla filtrow Czebyszewa i eliptycznego
rp = 1             # tetnienia w pasmie przepustowym [dB]
rs = 20            # tlumienie w pasmie zaporowym [dB]


def db(H):
    # Zamiana charakterystyki amplitudowej na decybele
    return 20 * np.log10(np.abs(H) + 1e-12)


# Lista filtrow do narysowania.
# Kazdy element ma: nazwe, wspolczynniki licznika b, wspolczynniki mianownika a.
filtry = []

# Klasyczne filtry NOI/IIR rzedu 3.
b, a = signal.butter(N, fg, btype='low', fs=fs)
filtry.append(('Butterworth, N=3', b, a))

b, a = signal.cheby1(N, rp, fg, btype='low', fs=fs)
filtry.append(('Czebyszew I, N=3', b, a))

b, a = signal.cheby2(N, rs, fg, btype='low', fs=fs)
filtry.append(('Czebyszew II, N=3', b, a))

b, a = signal.ellip(N, rp, rs, fg, btype='low', fs=fs)
filtry.append(('Eliptyczny, N=3', b, a))

# Jedno okno graficzne z dwoma wykresami: amplituda i faza.
fig, ax = plt.subplots(2, 1, figsize=(12, 9))

for nazwa, b, a in filtry:
    # Obliczenie charakterystyki czestotliwosciowej filtru
    f, H = signal.freqz(b, a, worN=2048, fs=fs)

    amplituda = db(H)
    faza = np.unwrap(np.angle(H)) * 180 / np.pi

    ax[0].plot(f, amplituda, linewidth=1.8, label=nazwa)
    ax[1].plot(f, faza, linewidth=1.8, label=nazwa)

ax[0].set_title('Charakterystyki amplitudowe filtrow dolnoprzepustowych')
ax[0].set_xlabel('Czestotliwosc [Hz]')
ax[0].set_ylabel('Amplituda [dB]')
ax[0].legend(ncol=4, fontsize=9)
ax[0].set_ylim(-60, 5)
ax[0].set_xlim(0, 4000)

ax[1].set_title('Charakterystyki fazowe filtrow dolnoprzepustowych')
ax[1].set_xlabel('Czestotliwosc [Hz]')
ax[1].set_ylabel('Faza [stopnie]')
ax[1].legend(ncol=4, fontsize=9)
ax[1].set_xlim(0, 4000)

for wykres in ax:
    wykres.grid(True, linestyle='--', alpha=0.4)

fig.suptitle('Porownanie filtrow dolnoprzepustowych, fs = 8 kHz, fg = 500 Hz', fontsize=14)
plt.tight_layout(rect=[0, 0.07, 1, 0.95])
plt.show()