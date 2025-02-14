from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Build a simple MLP model
mlp_model = Sequential([
    Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3)

mlp_history = mlp_model.fit(
    X_train_scaled, y_train, 
    validation_split=0.2, 
    epochs=20, 
    callbacks=[early_stop],
    verbose=1
)

loss, accuracy = mlp_model.evaluate(X_test_scaled, y_test)
print("MLP Test Accuracy:", accuracy)
