import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error
import joblib
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2



# WCZYTANIE I CZYSZCZENIE DANYCH
print("Szukam plików CSV z danymi...")
lista_plikow = glob.glob('Pliki .csv/*.csv') #wczytanie plików symulacyjnych
if not lista_plikow:
    print("BŁĄD: Nie znaleziono żadnych plików CSV w folderze!")
    exit()

tabela_tymczasowa = []
for plik in lista_plikow:
    temp_df = pd.read_csv(plik, header=None)
    tabela_tymczasowa.append(temp_df) #sklejanie plików

df = pd.concat(tabela_tymczasowa, ignore_index=True)
df = df.drop_duplicates() #usuwanie duplikatów
print(f"Wczytano i sklejono pliki. Unikalnych symulacji w bazie: {len(df)}") 

X = df.iloc[:, 0:3].values
y = df.iloc[:, 3:5].values

# PODZIAŁ I PODWÓJNE SKALOWANIE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# BUDOWA SIECI
print("\nBudowa i uczenie modelu...")

#06_06 - zwiększono liczbę neuronów głębszych warstw oraz zmieniono funkcję aktywacji
model = Sequential([
    Dense(256, activation='swish',kernel_regularizer=l2(1e-4), input_shape=(3,)),
    Dense(128, activation='swish',kernel_regularizer=l2(1e-4)),
    Dense(64, activation='swish',kernel_regularizer=l2(1e-4)),
    Dense(32, activation='swish',kernel_regularizer=l2(1e-4)),
    Dense(2, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

# TRENOWANIE
early_stop = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(
    X_train_scaled, y_train_scaled, 
    epochs=500, 
    batch_size=32, 
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=0 
)

# 5. EWALUACJA
print("\nKoniec uczenia! Ocena na zbiorze testowym (ślepa próba):")
y_pred_scaled = model.predict(X_test_scaled)
y_pred_real = scaler_y.inverse_transform(y_pred_scaled)

mae_s431 = mean_absolute_error(y_test[:, 0], y_pred_real[:, 0])
mae_zysk = mean_absolute_error(y_test[:, 1], y_pred_real[:, 1])

print(f"Średni błąd bezwzględny dla S431: {mae_s431:.3f} dB")
print(f"Średni błąd bezwzględny dla Zysku: {mae_zysk:.3f} dB")

# WIZUALIZACJA WYNIKÓW 
plt.figure(figsize=(18, 5))

# Krzywa Loss
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Zbiór uczący', linewidth=2)
plt.plot(history.history['val_loss'], label='Zbiór walidacyjny', linewidth=2)
plt.title('Zbieżność procesu uczenia (MSE)', fontsize=14)
plt.xlabel('Liczba epok', fontsize=12)
plt.ylabel('Znormalizowany błąd średniokwadratowy', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)

# Predykcja S431 
plt.subplot(1, 3, 2)
plt.scatter(y_test[:, 0], y_pred_real[:, 0], alpha=0.6, edgecolors='k', color='royalblue', label='Punkty predykcji')
min_val_s = min(np.min(y_test[:, 0]), np.min(y_pred_real[:, 0]))
max_val_s = max(np.max(y_test[:, 0]), np.max(y_pred_real[:, 0]))
plt.plot([min_val_s, max_val_s], [min_val_s, max_val_s], 'r--', lw=2, label='Charakterystyka idealna (y=x)')
plt.title('Zdolność predykcyjna: Przesłuch $S_{431}$', fontsize=14)
plt.xlabel('Wartość referencyjna (OpenEMS) [dB]', fontsize=12)
plt.ylabel('Wartość estymowana przez sieć [dB]', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)

# Predykcja Zysku 
plt.subplot(1, 3, 3)
plt.scatter(y_test[:, 1], y_pred_real[:, 1], alpha=0.6, edgecolors='k', color='forestgreen', label='Punkty predykcji')
min_val_z = min(np.min(y_test[:, 1]), np.min(y_pred_real[:, 1]))
max_val_z = max(np.max(y_test[:, 1]), np.max(y_pred_real[:, 1]))
plt.plot([min_val_z, max_val_z], [min_val_z, max_val_z], 'r--', lw=2, label='Charakterystyka idealna (y=x)')
plt.title('Zdolność predykcyjna: Zysk przetwarzania', fontsize=14)
plt.xlabel('Wartość referencyjna (OpenEMS) [dB]', fontsize=12)
plt.ylabel('Wartość estymowana przez sieć [dB]', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ZAPIS MODELU I SKALERÓW DO APLIKACJI
model.save('model_pcb_grupa15_PRO.keras')
print("\nZapisano ulepszony model jako 'model_pcb_grupa15_PRO.keras'")
joblib.dump(scaler_X, 'skaler_x_pro.save')
print("Zapisano skaler wejściowy jako 'skaler_x_pro.save'")
joblib.dump(scaler_y, 'skaler_y_pro.save')
print("Zapisano skaler wyjściowy jako 'skaler_y_pro.save'")