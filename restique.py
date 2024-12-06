import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, BatchNormalization, LeakyReLU, Dropout, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Mount Google Drive (if using Google Colab)
from google.colab import drive
drive.mount('/content/drive')

# Load and prepare data
def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path, header=0)
    X = data.iloc[:, 0:57].values.astype(float)
    Y = data.iloc[:, 57].values

    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    return X_normalized, Y

# Load data for both electrodes
X_f3, Y_f3 = load_and_prepare_data('/content/drive/MyDrive/Dataset_Maker/Datacenter/Datacenter12.csv')
X_f4, Y_f4 = load_and_prepare_data('/content/drive/MyDrive/Dataset_Maker/Datacenter/Datacenter14.csv')

# Ensure Y_f3 and Y_f4 are identical (same labels for both electrodes)
assert np.array_equal(Y_f3, Y_f4), "Labels for F3 and F4 should be identical"

# Encode labels
encoder = LabelEncoder()
encoder_Y = encoder.fit_transform(Y_f3)  # Use Y_f3 or Y_f4, they should be the same
dummy_Y = to_categorical(encoder_Y)

# Split data
X_f3_train, X_f3_test, X_f4_train, X_f4_test, Y_train, Y_test = train_test_split(
    X_f3, X_f4, dummy_Y, test_size=0.2, random_state=42)

# Reshape data for Conv1D
X_f3_train = X_f3_train.reshape(X_f3_train.shape[0], 57, 1)
X_f3_test = X_f3_test.reshape(X_f3_test.shape[0], 57, 1)
X_f4_train = X_f4_train.reshape(X_f4_train.shape[0], 57, 1)
X_f4_test = X_f4_test.reshape(X_f4_test.shape[0], 57, 1)

def build_combined_cnn_model(input_shape, num_classes):
    def create_branch(inputs):
        x = Conv1D(64, kernel_size=3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = Conv1D(128, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = Conv1D(256, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = Conv1D(512, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = Conv1D(1024, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = GlobalMaxPooling1D()(x)
        return x

    # F3 input branch
    input_f3 = Input(shape=input_shape)
    x_f3 = create_branch(input_f3)

    # F4 input branch
    input_f4 = Input(shape=input_shape)
    x_f4 = create_branch(input_f4)

    # Combine F3 and F4 branches
    combined = Concatenate()([x_f3, x_f4])

    x = Dense(768, kernel_regularizer=l2(0.0001))(combined)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.4)(x)

    x = Dense(384, kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.4)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[input_f3, input_f4], outputs=outputs)
    return model

# Build and compile model
model = build_combined_cnn_model((57, 1), len(np.unique(Y_f3)))
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)

# Train model
history = model.fit(
    [X_f3_train, X_f4_train], Y_train,
    validation_data=([X_f3_test, X_f4_test], Y_test),
    epochs=150,
    batch_size=64,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate model
loss, accuracy = model.evaluate([X_f3_test, X_f4_test], Y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Predict and evaluate
Y_pred = model.predict([X_f3_test, X_f4_test])
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true_classes = np.argmax(Y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(Y_true_classes, Y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Classification Report
class_names = list(encoder.classes_)
print('Classification Report')
print(classification_report(Y_true_classes, Y_pred_classes, target_names=class_names))
