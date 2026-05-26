from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


plik = Path(__file__).parent / "chlopak.JPG"
obraz = Image.open(plik).convert("RGB")

histogram = obraz.histogram()

red = histogram[0:256]
green = histogram[256:512]
blue = histogram[512:768]
poziomy = range(256)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(obraz)
plt.title("Obraz chlopak.JPG")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.fill_between(poziomy, red, color="red", alpha=0.6, label="Kanal Red")
plt.fill_between(poziomy, green, color="green", alpha=0.6, label="Kanal Green")
plt.fill_between(poziomy, blue, color="blue", alpha=0.6, label="Kanal Blue")
plt.title("Histogram obrazu chlopak.JPG")
plt.xlabel("Intensywnosc piksela (0-255)")
plt.ylabel("Czestotliwosc")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.xlim(0, 255)

plt.show()
