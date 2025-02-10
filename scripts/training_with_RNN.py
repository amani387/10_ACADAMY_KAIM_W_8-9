from tensorflow.keras.layers import SimpleRNN, LSTM

# Simple RNN Model
rnn_model = Sequential([
    SimpleRNN(32, input_shape=(X_train_scaled.shape[1], 1)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn_history = rnn_model.fit(
    X_train_cnn, y_train,
    validation_split=0.2,
    epochs=10,
    callbacks=[early_stop],
    verbose=1
)
loss, accuracy = rnn_model.evaluate(X_test_cnn, y_test)
print("RNN Test Accuracy:", accuracy)

# LSTM Model
lstm_model = Sequential([
    LSTM(32, input_shape=(X_train_scaled.shape[1], 1)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_history = lstm_model.fit(
    X_train_cnn, y_train,
    validation_split=0.2,
    epochs=10,
    callbacks=[early_stop],
    verbose=1
)
loss, accuracy = lstm_model.evaluate(X_test_cnn, y_test)
print("LSTM Test Accuracy:", accuracy)
