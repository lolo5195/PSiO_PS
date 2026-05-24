import matplotlib.pyplot as plt
import numpy as np

def spectrogram(x, fs, tytul, ax, dlugosc_okna=256, krok=128, alfa=0.5):
    okno = np.hamming(dlugosc_okna)
    starty = np.arange(0, len(x) - dlugosc_okna + 1, krok)

    czestotliwosci = np.fft.rfftfreq(dlugosc_okna, 1 / fs)
    czasy = (starty + dlugosc_okna / 2) / fs
    periodogramy = []

    for start in starty:
        ramka = x[start:start + dlugosc_okna] * okno
        widmo = np.fft.rfft(ramka)
        moc = np.abs(widmo) ** 2 / (fs * np.sum(okno ** 2))
        moc[1:-1] = 2 * moc[1:-1]
        periodogramy.append(moc)

    periodogramy = np.array(periodogramy).T
    wygladzone = np.zeros_like(periodogramy)

    wygladzone[:, 0] = periodogramy[:, 0]
    for l in range(1, len(czasy)):
        # Wzor z zadania 4.1: P = alfa * P_poprzednie + (1 - alfa) * |X(k,l)|^2
        wygladzone[:, l] = alfa * wygladzone[:, l - 1] + (1 - alfa) * periodogramy[:, l]

    gestosc_mocy_db = 10 * np.log10(wygladzone + 1e-20)
    T, F = np.meshgrid(czasy, czestotliwosci)

    ax.plot_surface(T, F, gestosc_mocy_db, cmap='viridis',
                    edgecolor='black', linewidth=0.15, rstride=3, cstride=1)
    ax.set_title(tytul + f', alfa = {alfa}')
    ax.set_xlabel('Czas [s]', labelpad=8)
    ax.set_ylabel('Czestotliwosc [Hz]', labelpad=8)
    ax.set_zlabel('PSD [dB/Hz]', labelpad=12)
    ax.set_ylim(0, fs / 2)
    ax.view_init(elev=25, azim=-135)
    ax.tick_params(labelsize=8)

    return czasy, czestotliwosci, gestosc_mocy_db


fs = 8000
N = 4096
f1 = 500
f2 = 1200
t = np.arange(N) / fs

szum = np.random.normal(0, 1, N)
sygnal = 0.5 * np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
sygnal_z_szumem = sygnal + 0.1 * szum

fig = plt.figure(figsize=(20, 7))
fig.suptitle('usredniane periodogramy')

ax1 = fig.add_subplot(1, 3, 1, projection='3d')
spectrogram(szum, fs, 'Szum gaussowski', ax1)

ax2 = fig.add_subplot(1, 3, 2, projection='3d')
spectrogram(sygnal, fs, 'Dwie sinusoidy', ax2)

ax3 = fig.add_subplot(1, 3, 3, projection='3d')
spectrogram(sygnal_z_szumem, fs, 'Sinusoidy z szumem', ax3)

plt.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.88, wspace=0.15)
plt.show()