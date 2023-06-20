from models import Attention_ResUNet
from losses import dice_coef_loss, jacard_coef_loss
import os
import PIL
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

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

# Save the plot to a file
plot_path = os.path.join('C:\\Users\\yunus\\PycharmProjects\\AttentionResUNet\\', 'slices_plot.png')
plt.savefig(plot_path)
plt.close()

###############Training##############################

# Convert the image and mask lists to numpy arrays
image_array = np.array(image_list)
mask_array = np.array(mask_list)

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(image_array, mask_array, test_size=0.2, random_state=42)

# Define the model
model = Attention_ResUNet(input_shape=(128, 128, 1))  # Instantiate the Attention ResUNet model

# Define the loss functions
jaccard_loss = jacard_coef_loss
dice_loss = dice_coef_loss

# Define the optimizer and metrics
optimizer = Adam(learning_rate=0.001)

metrics = ['accuracy', jaccard_loss, dice_loss]  # Add jaccard_loss and dice_loss as metrics

# Compile the model
model.compile(optimizer=optimizer, loss=[jaccard_loss, dice_loss], metrics=metrics)

# Train the model
epochs = 2
batch_size = 16

callbacks = [
    TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
]

# Train the model
losses = []
jaccard_losses = []
dice_losses = []
accuracies = []

for epoch in range(epochs):
    # Perform one epoch of training
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=1, callbacks=callbacks)

    # Make predictions on the training set
    y_pred_train = model.predict(X_train)

    # Calculate the metrics
    loss = model.evaluate(X_train, y_train, verbose=0)
    jaccard_loss_value = jacard_coef_loss(y_train, y_pred_train)
    dice_loss_value = dice_coef_loss(y_train, y_pred_train)
    accuracy = accuracy_score(y_train.flatten(), y_pred_train.flatten().round())

    # Append the metric values to the respective lists
    losses.append(loss)
    jaccard_losses.append(jaccard_loss_value)
    dice_losses.append(dice_loss_value)
    accuracies.append(accuracy)

# Plot training history
plt.figure()
plt.plot(losses, label='Training Loss')
plt.plot(jaccard_losses, label='Training Jaccard Loss')
plt.plot(dice_losses, label='Training Dice Loss')
plt.plot(accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Save the training history plot to a file
history_plot_path = os.path.join('C:\\Users\\yunus\\PycharmProjects\\AttentionResUNet\\', 'training_history.png')
plt.savefig(history_plot_path)
plt.close()

###############Evaluation##############################

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Jaccard and Dice scores on the test set
jaccard_score = jacard_coef_loss(y_test, y_pred)
dice_score = dice_coef_loss(y_test, y_pred)

# Flatten the arrays
y_test_flattened = (y_test / 255).flatten()
y_pred_flattened = (y_pred / 255).flatten()

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test_flattened, y_pred_flattened)
auc = roc_auc_score(y_test_flattened, y_pred_flattened)

# Plot ROC curve and save to file
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.show()

# Save the ROC plot to a file
roc_plot_path = os.path.join('C:\\Users\\yunus\\PycharmProjects\\AttentionResUNet\\', 'roc_plot.png')
plt.savefig(roc_plot_path)
plt.close()
