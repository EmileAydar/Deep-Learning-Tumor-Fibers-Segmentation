from models import Attention_ResUNet
from losses import dice_coef, jacard_coef
import os
import PIL
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import tensorflow as tf
from keras.optimizers import Adam

###############Data Loader##############################

image_path = os.path.join('C:\\Users\\yunus\\PycharmProjects\\AttentionResUNet\\image_test\\patch_0_0_0.tif')
mask_path = os.path.join('C:\\Users\\yunus\\PycharmProjects\\AttentionResUNet\\mask_test\\patch_0_0_0_finalprediction.ome.tif')

image_list = []
mask_list = []

image = tifffile.imread(image_path)
mask = tifffile.imread(mask_path)

# Iterate over each slice of the image and mask
for i in range(image.shape[0]):
    image_slice = image[i]
    mask_slice = mask[i]

    # Append the slices to the respective lists
    image_list.append(image_slice)
    mask_list.append(mask_slice)

# Display the first few slices
num_slices_to_display = 5
for i in range(num_slices_to_display):
    plt.subplot(2, num_slices_to_display, i+1)
    plt.imshow(image_list[i], cmap='gray')
    plt.title(f'Image Slice {i+1}')
    plt.axis('off')

    plt.subplot(2, num_slices_to_display, num_slices_to_display+i+1)
    plt.imshow(mask_list[i], cmap='gray')
    plt.title(f'Mask Slice {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()

###############Training##############################

# Convert the image and mask lists to numpy arrays
image_array = np.array(image_list)
mask_array = np.array(mask_list)

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(image_array, mask_array, test_size=0.2, random_state=42)

# Define the model
model = Attention_ResUNet(input_shape=(128,128,1))  # Instantiate the Attention ResUNet model

# Define the loss functions
jaccard_loss = jacard_coef
dice_loss = dice_coef

# Define the optimizer and metrics
optimizer = tf.keras.optimizers.Adam()

metrics = ['accuracy', jaccard_loss, dice_loss]  # Add jaccard_loss and dice_loss as metrics

# Compile the model
model.compile(optimizer='Adam', loss=[jaccard_loss, dice_loss], metrics=metrics)

# Train the model
epochs = 2

history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

# Plot the training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history.history['dice_coef'], label='Dice Score')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['jaccard_coef'], label='Jaccard Score')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


###############Evaluation##############################

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Jaccard and Dice scores on the test set
jaccard_score = jacard_coef(y_test, y_pred)
dice_score = dice_coef(y_test, y_pred)

# Flatten the arrays
y_test_flattened = (y_test/255).flatten()
y_pred_flattened = (y_pred/255).flatten()

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test_flattened, y_pred_flattened)
auc = roc_auc_score(y_test_flattened, y_pred_flattened)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.show()
