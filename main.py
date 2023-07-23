import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from losses import dice_loss, tversky_loss, iou_coef, precision, recall, f1_score
from models import AttentionResUNet, ThreeDAttentionResUNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import tifffile
import subprocess as sp
#########GPU Manager####################################################################################################
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.current_device()
torch.cuda.device(0)
torch.cuda.get_device_name(0)

def mask_unused_gpus(leave_unmasked=1):
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"

    try:
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        available_gpus = [i for i, x in enumerate(memory_free_values) if x > ACCEPTABLE_AVAILABLE_MEMORY]

        if len(available_gpus) < leave_unmasked:
            raise ValueError('Found only %d usable GPUs in the system' % len(available_gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, available_gpus[:leave_unmasked]))
    except Exception as e:
        print('"nvidia-smi" is probably not installed. GPUs are not masked', e)

# Define the device for training
num_gpus = torch.cuda.device_count()
if num_gpus >= 2:
    device = torch.device('cuda')
    device_ids = [0, 1]  # or whichever IDs your GPUs have
elif num_gpus == 1:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
#####################Data Loader########################################################################################
image_folder = 'Images'
mask_folder = 'Labels'

image_list = []
mask_list = []

# Load images from the image folder
for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    image = tifffile.imread(image_path)
    image_list.append(image)

# Load masks from the mask folder
for filename in os.listdir(mask_folder):
    mask_path = os.path.join(mask_folder, filename)
    mask = tifffile.imread(mask_path)
    mask_list.append(mask)

# Convert the image and mask lists to numpy arrays
image_array = np.array(image_list)
mask_array = np.array(mask_list)

### Reshape the image and mask arrays
## 2D Reshaping
#image_array = image_array.reshape((-1, 1, 128, 128))
#mask_array = mask_array.reshape((-1, 1, 128, 128))

## 3D Reshaping
image_array = image_array.reshape((-1, 1, 256, 128, 128))
mask_array = mask_array.reshape((-1, 1,256, 128, 128))

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(image_array, mask_array, test_size=0.2, random_state=42)

# Convert data to torch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# Create data loaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)
###########################Training#####################################################################################
# Define the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ThreeDAttentionResUNet(input_channels=1, output_channels=1)
if torch.cuda.is_available():
    if num_gpus >= 2:
        model = nn.DataParallel(model, device_ids=[0,1])
    model = model.to(device)

# Define the optimizer
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Define the loss function
criterion = dice_loss
epochs = 100

# Training-Evaluation loop
for epoch in range(epochs):
    model.train()
    total_loss, total_iou, total_prec, total_rec, total_f1 = 0, 0, 0, 0, 0
    for i, data in enumerate(train_loader):
        # Get the inputs and labels
        inputs, labels = data

        # Move the inputs and labels to GPU if available
        if torch.cuda.is_available():
            inputs = inputs.to(device)
            labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # Calculate the loss
        loss = criterion(labels, outputs)

        # Calculate metrics
        iou = iou_coef(labels, outputs)
        prec = precision(labels, outputs)
        rec = recall(labels, outputs)
        f1 = f1_score(labels, outputs)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update total loss and metrics
        total_loss += loss.item()
        total_iou += iou.item()
        total_prec += prec.item()
        total_rec += rec.item()
        total_f1 += f1.item()

    # Print average loss and metrics for this epoch
    print('Epoch [{}/{}], Loss: {:.4f}, IoU: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'
          .format(epoch+1, epochs, total_loss / len(train_loader), total_iou / len(train_loader),
                  total_prec / len(train_loader), total_rec / len(train_loader), total_f1 / len(train_loader)))

    # Validation
    model.eval()
    total_val_loss = 0
    total_val_iou = 0
    total_val_prec = 0
    total_val_rec = 0
    total_val_f1 = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # Get the inputs and labels
            inputs, labels = data

            # Move the inputs and labels to GPU if available
            if torch.cuda.is_available():
                inputs = inputs.to(device)
                labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            val_loss = criterion(labels, outputs)

            # Calculate metrics
            total_val_iou += iou_coef(labels, outputs).item()
            total_val_prec += precision(labels, outputs).item()
            total_val_rec += recall(labels, outputs).item()
            total_val_f1 += f1_score(labels, outputs).item()
            total_val_loss += val_loss.item()

    # Print average validation loss and metrics for this epoch
    print(
        'Epoch [{}/{}], Validation Loss: {:.4f}, Val IoU: {:.4f}, Val Precision: {:.4f}, Val Recall: {:.4f}, Val F1: {:.4f}'
        .format(epoch + 1, epochs, total_val_loss / len(test_loader), total_val_iou / len(test_loader),
                total_val_prec / len(test_loader), total_val_rec / len(test_loader), total_val_f1 / len(test_loader)))

# Save the model
torch.save(model.module.state_dict() if num_gpus > 1 else model.state_dict(), 'model_weights.pth')

###########################Evaluation###################################################################################
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

model.eval()

# Initialize lists for true and predicted labels
true_labels = []
pred_labels = []

# Initialize metrics
total_test_loss = 0
total_test_iou = 0
total_test_prec = 0
total_test_rec = 0
total_test_f1 = 0

# Iterate over the test set
for inputs, labels in test_loader:
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()
    with torch.no_grad():
        outputs = model(inputs)

    test_loss = criterion(labels, outputs).item()
    total_test_loss += test_loss
    total_test_iou += iou_coef(labels, outputs).item()
    total_test_prec += precision(labels, outputs).item()
    total_test_rec += recall(labels, outputs).item()
    total_test_f1 += f1_score(labels, outputs).item()

    outputs = outputs.cpu().numpy() > 0.5
    labels = labels.cpu().numpy() > 0.5

    true_labels.extend(labels.reshape(-1))
    pred_labels.extend(outputs.reshape(-1))

# Print average test loss and metrics
print('Test Loss: {:.4f}, Test IoU: {:.4f}, Test Precision: {:.4f}, Test Recall: {:.4f}, Test F1: {:.4f}'
      .format(total_test_loss / len(test_loader), total_test_iou / len(test_loader),
              total_test_prec / len(test_loader), total_test_rec / len(test_loader), total_test_f1 / len(test_loader)))

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(true_labels, pred_labels)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_plot.png')
plt.show()



