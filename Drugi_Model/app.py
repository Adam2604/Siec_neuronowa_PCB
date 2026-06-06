import sys
import joblib
import numpy as np
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QSlider, QGroupBox, QFormLayout)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush
from tensorflow.keras.models import load_model 

# WIDŻET DO RYSOWANIA PŁYTKI
class PCBVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(500, 300)
        self.setStyleSheet("background-color: #2b2b2b;")
        self.dist, self.sep, self.shift = 500, 300, 2000 #parametry startowe

    def update_parameters(self, dist, sep, shift):
        self.dist, self.sep, self.shift = dist, sep, shift
        self.update()

    #metoda do rysowania
    def paintEvent(self, event):
        #przygotowanie narzędzia do rysowania
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        #skala i punkty bazowe do rysowania
        sc, ox, oy = 1/65.0, 20, 80 #skalowanie, offsety
        cx, cy, th = ox + 30000*sc/2, oy + 12000*sc/2, max(3, 500*sc) #centrum płytki, grubość ścieżek (nie da się mniej niż 3px)

        #rysowanie laminatu PCB
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(90, 135, 145))
        p.drawRect(ox, oy, int(30000*sc), int(12000*sc))

        #rysowanie głównej ścieżki
        p.setPen(QPen(QColor(160, 95, 125), 1))
        p.setBrush(QColor(160, 95, 125))
        p.drawRect(int(cx), int(cy - th/2), int(15000*sc), int(th))
        
        #rysowanie przelotki
        vr = max(3, 250*sc) #promień przelotki
        p.setBrush(QColor(50, 50, 200))
        p.drawEllipse(int(cx - vr), int(cy - vr), int(vr*2), int(vr*2))

        #rysowanie pary różnicowej
        p.setBrush(QColor(160, 95, 125))
        #liczyby lewej krawędzi ścieżki (od max długości odejmujemy przesunięcie, a następnie długośc samej ścieżki)
        diff_x = ox + (30000 - self.shift - 19500) * sc 
        y_base = cy - th/2 + th  #poziom zero na osi pionowej dla ściezek dolnych
        
        #pętla dla przesunięć suwaków
        for y_offset in [self.dist, self.dist + 500 + self.sep]:
            p.drawRect(int(diff_x), int(y_base + y_offset*sc), int(19500*sc), int(th))

        #napisy z ustawieniami
        p.setPen(QColor(150, 150, 150))
        p.drawText(10, 20, 400, 100, Qt.AlignmentFlag.AlignLeft, 
                   f"Wizualizacja PCB 2D\nDistance: {self.dist} | Sep: {self.sep} | Shift: {self.shift}")

# GŁÓWNE OKNO APLIKACJI
class PCBSimulatorApp(QWidget):
    def __init__(self):
        super().__init__()
        try:
            self.model = load_model('model_pcb_grupa15_PRO.keras') # Ładowanie modelu
            self.scaler = joblib.load('skaler_x_pro.save') # Ładowanie prawdziwego skalera X
            self.scaler_y = joblib.load('skaler_y_pro.save') # Skaler Y
            self.model_loaded = True
        except Exception as e:
            print(f"Błąd ładowania: {e}")
            self.model_loaded = False

        self.initUI()

    def initUI(self):
        self.setWindowTitle('MN - Symulator Przeników PCB - Grupa 15')
        self.resize(900, 400)
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        
        #layout dla suwaków i przewidywań
        controls_group = QGroupBox("Parametry wejściowe (um)")
        form_layout = QFormLayout()

        self.slider_dist = self.create_slider(200, 2000, 500)
        self.label_dist_val = QLabel("500")
        form_layout.addRow("Distance:", self.create_row(self.slider_dist, self.label_dist_val))

        self.slider_sep = self.create_slider(200, 400, 300)
        self.label_sep_val = QLabel("300")
        form_layout.addRow("Separation:", self.create_row(self.slider_sep, self.label_sep_val))

        self.slider_shift = self.create_slider(0, 10000, 2000)
        self.label_shift_val = QLabel("2000")
        form_layout.addRow("Shift from Edge:", self.create_row(self.slider_shift, self.label_shift_val))

        controls_group.setLayout(form_layout)
        left_layout.addWidget(controls_group)

        results_group = QGroupBox("Przewidywania Sieci Neuronowej")
        res_layout = QVBoxLayout()
        self.label_s431 = QLabel("S431_dB_av: --- dB")
        self.label_s431.setStyleSheet("font-size: 16px; font-weight: bold; color: #1f77b4;")
        self.label_gain = QLabel("Diff gain: --- dB")
        self.label_gain.setStyleSheet("font-size: 16px; font-weight: bold; color: #2ca02c;")
        res_layout.addWidget(self.label_s431)
        res_layout.addWidget(self.label_gain)
        results_group.setLayout(res_layout)
        left_layout.addWidget(results_group)
        left_layout.addStretch()

        #dodanie layoutów
        self.canvas = PCBVisualizer()
        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(self.canvas, 2)
        self.setLayout(main_layout)
        
        #przypisanie sliderom funkcji
        self.slider_dist.valueChanged.connect(self.update_app)
        self.slider_sep.valueChanged.connect(self.update_app)
        self.slider_shift.valueChanged.connect(self.update_app)
        self.update_app()

    def create_slider(self, min_val, max_val, default_val):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        return slider

    def create_row(self, slider, label):
        row = QHBoxLayout()
        row.addWidget(slider)
        row.addWidget(label)
        return row

    def update_app(self):
        val_dist, val_sep, val_shift = self.slider_dist.value(), self.slider_sep.value(), self.slider_shift.value()
        self.label_dist_val.setText(str(val_dist))
        self.label_sep_val.setText(str(val_sep))
        self.label_shift_val.setText(str(val_shift))
        self.canvas.update_parameters(val_dist, val_sep, val_shift)
        
        if self.model_loaded:
            # Użycie prawdziwego skalera z procesu uczenia
            input_data = np.array([[val_dist, val_sep, val_shift]])
            scaled_input = self.scaler.transform(input_data)
            scaled_prediction = self.model.predict(scaled_input, verbose=0)  
            prediction = self.scaler_y.inverse_transform(scaled_prediction)

            self.label_s431.setText(f"S431_dB_av: {prediction[0][0]:.3f} dB")
            self.label_gain.setText(f"Diff gain: {prediction[0][1]:.3f} dB")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PCBSimulatorApp()
    ex.show()
    sys.exit(app.exec())