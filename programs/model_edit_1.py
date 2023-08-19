import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def create_model(image_size, num_classes):
    base_model = tf.keras.applications.InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size[0], image_size[1], 3)
    )
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    return model

def train_model(model, X_train, y_train, X_val, y_val, batch_size, epochs):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val)
    )

    return history

def plot_accuracy_loss(history):
    # Plot the training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('accuracy_loss.png')
    plt.show()

    # Plot the training and validation accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('confusion_matrix.png')
    plt.show()

# Load the preprocessed data
X = np.load("data\preprocessed_data.npy")
y = np.load("data\preprocessed_labels.npy")

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the labels to one-hot encoded vectors
num_classes = len(np.unique(y))
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)

# Create the model
image_size = (224, 224)  # Set the desired image size
model = create_model(image_size, num_classes)

# Train the model
batch_size = 32
epochs = 20
history = train_model(model, X_train, y_train, X_val, y_val, batch_size, epochs)

# Save the trained model
model.save("bird_classifier_model.h5")

# Plot accuracy and loss
plot_accuracy_loss(history)

# Evaluate the model on the validation set
y_val_pred = model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)
y_val_true_classes = np.argmax(y_val, axis=1)

# Print classification report
class_labels = [str(i) for i in range(num_classes)]
print(classification_report(y_val_true_classes, y_val_pred_classes, target_names=class_labels))

# Calculate F1 score
f1 = f1_score(y_val_true_classes, y_val_pred_classes, average='weighted')
print("F1 Score:", f1)

# Plot confusion matrix
plot_confusion_matrix(y_val_true_classes, y_val_pred_classes, class_labels)
