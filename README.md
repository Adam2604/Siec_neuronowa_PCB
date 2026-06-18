Projekt realizowany w ramach przedmiotu Metody Numeryczne (semestr 6, EiT). Głównym celem było zaprojektowanie i optymalizacja modelu zastępczego z wykorzystaniem sztucznych sieci neuronowych (ANN), który pozwala natychmiastowo oszacować uśredniony poziom przesłuchu ($S_{431}$) oraz różnicowy zysk przetwarzania na podstawie zmiennej geometrii ścieżek na płytce PCB. Stworzony model pozwala skrócić czas oczekiwania na wyniki z kilku minut (w przypadku tradycyjnych symulacji falowych) do ułamka sekundy.

**Główne cechy i technologie**
- Dataset: 3200 unikalnych symulacji elektromagnetycznych zrealizowanych metodą FDTD w środowisku MATLAB / OpenEMS.
- Sieć neuronowa (ANN): Wielowarstwowy perceptron (MLP) zaimplementowany w Pythonie (TensorFlow/Keras). Zoptymalizowany z wykorzystaniem funkcji aktywacji Swish, regularyzacji $L_2$ oraz adaptacyjnego kroku uczenia.
- Aplikacja GUI: Interaktywne okno stworzone w PyQt6 z wykorzystaniem klasy QPainter. Umożliwia dynamiczne zmienianie parametrów geometrycznych suwakami i podgląd predykcji sieci oraz renderowanego układu 2D w czasie rzeczywistym.
<img width="989" height="459" alt="image" src="https://github.com/user-attachments/assets/45ff7833-7341-4c1b-ba33-c2a3e607cf21" />

**Pliki projektu**

Plik **app.py**: Interaktywna aplikacja GUI

Plik **siec.py**: Skrypt generujący i optymalizujący sieć neuronową 

Plik **symulacje.m**: Symulacje przeprowadzone w MATLAB-ie w celu zebrania danych do wytrenowania sieci neuronowych

Plik **model_pcb_grupa15_PRO**: wytrenowany model sieci neuronowej
