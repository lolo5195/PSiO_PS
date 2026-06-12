import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# Współczynniki filtru: b = [b0, b1, b2], a = [1, a1, a2].
b = [0.5, 0.9, 0.3]
a = [1, 0.25, 0.9]


def filtr(x, b, a):
    y = np.zeros(len(x))

    for n in range(len(x)):
        y[n] = b[0] * x[n]

        if n >= 1:
            y[n] += b[1] * x[n - 1] - a[1] * y[n - 1]

        if n >= 2:
            y[n] += b[2] * x[n - 2] - a[2] * y[n - 2]

    return y


def na_db(wartosci):
    return 20 * np.log10(np.abs(wartosci) + 1e-12)


# a) Odpowiedź impulsowa i b) porównanie z funkcją filter/lfilter.
N = 100
impuls = np.zeros(N)
impuls[0] = 1
h_moja = filtr(impuls, b, a)
h_filter = signal.lfilter(b, a, impuls)

print('Czy odpowiedź impulsowa zgadza się z lfilter?', np.allclose(h_moja, h_filter))
print('Największa różnica:', np.max(np.abs(h_moja - h_filter)))

# c) Charakterystyka częstotliwościowa filtru.
w, H = signal.freqz(b, a, 1024)
f = w / np.pi
H_db = na_db(H)
faza_stopnie = np.unwrap(np.angle(H)) * 180 / np.pi

# d) Zera, bieguny i stabilność.
zera, bieguny, _ = signal.tf2zpk(b, a)
stabilny = np.all(np.abs(bieguny) < 1)

print('Zera:', zera)
print('Bieguny:', bieguny)
print('Moduły biegunów:', np.abs(bieguny))
print('Filtr stabilny:', stabilny)

# e) Szum gaussowski oraz widma amplitudowe.
np.random.seed(42)
x = np.random.normal(0, 1, 256)
y = filtr(x, b, a)

freq = 2 * np.fft.rfftfreq(len(x))
X_db = na_db(np.fft.rfft(x) / len(x))
Y_db = na_db(np.fft.rfft(y) / len(y))
_, H_fft = signal.freqz(b, a, freq * np.pi)
H_fft_db = na_db(H_fft)

# Wszystkie wykresy w jednym oknie graficznym.
fig, ax = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
odp_impulsowa, amplituda, faza, zera_bieguny, widma, porownanie = ax.flat

n = np.arange(N)
odp_impulsowa.stem(n, h_moja, basefmt=' ', linefmt='C0-', markerfmt='C0o', label='moja funkcja')
odp_impulsowa.plot(n, h_filter, 'C1--', linewidth=2, label='lfilter')
odp_impulsowa.set_title('a-b) Odpowiedź impulsowa')
odp_impulsowa.set_xlabel('n')
odp_impulsowa.set_ylabel('h[n]')
odp_impulsowa.legend()

amplituda.plot(f, H_db, linewidth=2)
amplituda.set_title('c) Charakterystyka amplitudowa')
amplituda.set_xlabel('częstotliwość [×π rad/próbkę]')
amplituda.set_ylabel('wzmocnienie [dB]')

faza.plot(f, faza_stopnie, color='C3', linewidth=2)
faza.set_title('c) Charakterystyka fazowa')
faza.set_xlabel('częstotliwość [×π rad/próbkę]')
faza.set_ylabel('faza [stopnie]')

kat = np.linspace(0, 2 * np.pi, 400)
zera_bieguny.plot(np.cos(kat), np.sin(kat), 'k--', linewidth=1)
zera_bieguny.scatter(zera.real, zera.imag, marker='o', s=90, facecolors='none', edgecolors='C0', linewidths=2)
zera_bieguny.scatter(bieguny.real, bieguny.imag, marker='x', s=90, color='C3', linewidths=2)

for i, zero in enumerate(zera, start=1):
    zera_bieguny.text(zero.real + 0.04, zero.imag + 0.04, f'zero {i}', color='C0')

for i, biegun in enumerate(bieguny, start=1):
    zera_bieguny.text(biegun.real + 0.04, biegun.imag + 0.04, f'biegun {i}', color='C3')

zera_bieguny.text(0.45, 0.88, 'okrąg jednostkowy', color='0.25')
zera_bieguny.axhline(0, color='0.5', linewidth=1)
zera_bieguny.axvline(0, color='0.5', linewidth=1)
zera_bieguny.set_aspect('equal', adjustable='box')
zera_bieguny.set_xlim(-1.45, 1.2)
zera_bieguny.set_ylim(-1.2, 1.2)
zera_bieguny.set_title('d) Zera i bieguny')
zera_bieguny.set_xlabel('Re')
zera_bieguny.set_ylabel('Im')

widma.plot(freq, X_db, label='x[n] - szum')
widma.plot(freq, Y_db, label='y[n] - po filtracji')
widma.set_title('e) Widma amplitudowe x[n] i y[n]')
widma.set_xlabel('częstotliwość [×π rad/próbkę]')
widma.set_ylabel('amplituda [dB]')
widma.legend()

# Charakterystykę filtru przesuwamy do poziomu widma szumu, żeby łatwiej porównać kształt.
porownanie.plot(freq, Y_db, label='widmo y[n]')
porownanie.plot(freq, np.median(X_db) + H_fft_db, 'C1--', linewidth=2, label='charakterystyka filtru')
porownanie.set_title('e) Widmo y[n] a charakterystyka filtru')
porownanie.set_xlabel('częstotliwość [×π rad/próbkę]')
porownanie.set_ylabel('amplituda [dB]')
porownanie.legend()

for wykres in ax.flat:
    wykres.grid(True, linestyle='--', alpha=0.4)

fig.suptitle('Analiza filtru cyfrowego', fontsize=16)
plt.show()
