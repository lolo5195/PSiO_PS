# Projekt: Klasyfikacja HIGGS (analiza danych, scikit-learn)

Akademicki projekt z analizy danych. Zadanie: **klasyfikacja binarna** na zbiorze **HIGGS** (czy zdarzenie to sygnal czy tlo) przy uzyciu klasycznych modeli scikit-learn. Cala praca w jednym notebooku Jupyter: analiza danych, feature selection i **30 eksperymentow** w 4 galeziach z roznymi zestawami cech.

Pelny plan realizacji: [PLAN_PROJEKTU.md](PLAN_PROJEKTU.md).

## Struktura projektu

```
projekt/
├── README.md                    # ten plik
├── requirements.txt             # zaleznosci (wersje bibliotek)
├── PLAN_PROJEKTU.md             # szczegolowy plan projektu
├── data/                        # tu umiesc HIGGS.csv.gz (NIE jest w repo)
├── higgs_analiza.ipynb          # glowny notebook (analiza + 30 eksperymentow)
├── wyniki/
│   ├── wyniki_eksperymentow.csv # tabela wynikow (generowana przez notebook)
│   └── wykresy/                 # wykresy PNG do raportu
└── dokumentacja/
    └── raport.pdf               # raport koncowy oddawany na CEZ
```

## 1. Srodowisko i instalacja bibliotek

Zalecany **Python >= 3.10**. Utworz srodowisko i zainstaluj zaleznosci:

```bash
# z katalogu projekt/
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Alternatywnie (conda):

```bash
conda create -n higgs python=3.11 numpy pandas matplotlib scikit-learn jupyterlab -y
conda activate higgs
```

Wymagane biblioteki (szczegoly i wersje minimalne w [requirements.txt](requirements.txt)): `numpy`, `pandas`, `matplotlib`, `scikit-learn` oraz `jupyterlab` / `ipykernel`.

## 2. Pobranie danych HIGGS

Zbior **nie jest** w repozytorium (plik ma ~2,6 GB po kompresji). Pobierz go do katalogu `data/`.

**Sposob A (zalecany) - bezposredni plik `.csv.gz`** (pandas czyta go bez rozpakowywania):

```bash
# z katalogu projekt/
curl -L -o data/HIGGS.csv.gz "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
```

**Sposob B - oficjalna strona UCI** (plik `higgs.zip`, w srodku `HIGGS.csv.gz`):

- Strona zbioru: https://archive.ics.uci.edu/dataset/280/higgs
- Pobierz `higgs.zip`, rozpakuj i umiesc `HIGGS.csv.gz` w `data/`.

Po pobraniu w `data/` powinien znalezc sie plik `HIGGS.csv.gz`.

### Informacje o zbiorze

- Klasyfikacja binarna: kolumna 0 = etykieta (1 = sygnal, 0 = tlo), kolumny 1-28 = cechy numeryczne.
- 28 cech = **21 niskopoziomowych** (kinematyka) + **7 wysokopoziomowych** (pochodne fizyczne).
- Plik jest **bez naglowka**; brak brakow danych; zbior niemal zbalansowany (~53% / ~47%).
- W projekcie wczytujemy tylko **~1,25 mln** pierwszych rekordow (parametr `nrows`), nie caly zbior (~11 mln). Po podziale 80/20 daje to **~1 mln rekordow treningowych**.

## 3. Uruchomienie

```bash
# z katalogu projekt/ (z aktywnym srodowiskiem)
jupyter lab        # lub: jupyter notebook
```

Otworz `higgs_analiza.ipynb` i uruchom komorki po kolei (Run All). Wyniki zapisuja sie do `wyniki/wyniki_eksperymentow.csv`, a wykresy do `wyniki/wykresy/`.

## 4. Reprodukowalnosc

Aby inny zespol uzyskal **te same wyniki**:

- Wszedzie ustawiony jest staly `random_state = 42` (podzial danych i modele).
- Wczytywana liczba rekordow jest stala: `nrows = 1_250_000`.
- Podzial: `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
- **Zapisz dokladne wersje bibliotek** i dolacz je do raportu:

```bash
pip freeze > dokumentacja/wersje_bibliotek.txt
```

W notebooku warto dodac komorke wypisujaca wersje (`python`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`), aby wersje byly widoczne bezposrednio w PDF.

## 5. Zrodlo danych

Baldi, P., Sadowski, P., Whiteson, D. *Searching for Exotic Particles in High-Energy Physics with Deep Learning.* Nature Communications 5 (2014).
Zbior: HIGGS Data Set, UCI Machine Learning Repository - https://archive.ics.uci.edu/dataset/280/higgs
