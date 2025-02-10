import numpy as np
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

# Reshape data: (samples, timesteps, channels)
X_train_cnn = np.expand_dims(X_train_scaled, axis=2)
X_test_cnn = np.expand_dims(X_test_scaled, axis=2)

cnn_model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_history = cnn_model.fit(
    X_train_cnn, y_train,
    validation_split=0.2,
    epochs=10,
    callbacks=[early_stop],
    verbose=1
)
loss, accuracy = cnn_model.evaluate(X_test_cnn, y_test)
print("CNN Test Accuracy:", accuracy)
