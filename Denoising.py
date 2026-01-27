import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

X = np.load("data/NIST_h_AAplant_int_5G_imagenes_snr_10.npy")
Y = np.load("data/NIST_h_AAplant_int_5G_imagenes.npy")
print(X.shape, Y.shape)

X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
Y_train, Y_val = train_test_split(Y, test_size=0.2, random_state=42)

def denoising_cnn(input_shape=(128,128,1)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)

    outputs = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model

model = denoising_cnn()
model.compile(
    optimizer='adam',
    loss='mse'
)

model.summary()

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=50,
    batch_size=16,
)

idx = 0
pred = model.predict(X_val[idx:idx+1])

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Ruidosa")
plt.imshow(X_val[idx].squeeze(), cmap='viridis')

plt.subplot(1,3,2)
plt.title("Reconstruida")
plt.imshow(pred.squeeze(), cmap='viridis')

plt.subplot(1,3,3)
plt.title("Limpia")
plt.imshow(Y_val[idx].squeeze(), cmap='viridis')

plt.show()

print("Entrada ruidosa:", X_val[idx].min(), X_val[idx].max(), X_val[idx].mean())
print("Predicción:", pred[idx].min(), pred[idx].max(), pred[idx].mean())
print("Objetivo:", Y_val[idx].min(), Y_val[idx].max(), Y_val[idx].mean())


