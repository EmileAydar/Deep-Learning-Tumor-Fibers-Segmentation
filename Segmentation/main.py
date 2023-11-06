# Emile Aydar
# Fiber Detection in Cold-Plasma treated lung tumors
# LPP/ISIR || Ecole Polytechnique/Sorbonne Université, 2023

# Segmentation Neural Nets (2D & 3D approach)

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from losses import dice_loss, tversky_loss, iou_coef, precision, recall, f1_score,accuracy
from models import AttentionResUNet, ThreeDAttentionResUNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import tifffile
import subprocess as sp
import matplotlib.pyplot as plt
#########GPU Manager####################################################################################################
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.current_device()
torch.cuda.device(0)
torch.cuda.get_device_name(0)

#If you want to mask gpus you won't use ;)
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

# defining the device for training
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

# data loading
for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    image = tifffile.imread(image_path)
    image_list.append(image)

# labeled masks loading
for filename in os.listdir(mask_folder):
    mask_path = os.path.join(mask_folder, filename)
    mask = tifffile.imread(mask_path)
    mask_list.append(mask)

image_array = np.array(image_list)
mask_array = np.array(mask_list)

### reshaping the image and mask arrays
## 2D Reshaping
image_array = image_array.reshape((-1, 1, 128, 128))
mask_array = mask_array.reshape((-1, 1, 128, 128))

## 3D Reshaping
#image_array = image_array.reshape((-1, 1, 256, 128, 128))
#mask_array = mask_array.reshape((-1, 1,256, 128, 128))

X_train, X_test, y_train, y_test = train_test_split(image_array, mask_array, test_size=0.2, random_state=42)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)
###########################Training#####################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttentionResUNet(input_channels=1, output_channels=1)
if torch.cuda.is_available():
    if num_gpus >= 2:
        model = nn.DataParallel(model, device_ids=[0,1])
    model = model.to(device)
    
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion = dice_loss
epochs = 20

train_losses=[]
val_losses=[]
train_iou=[]
val_iou=[]
train_acc=[]
val_acc=[]
train_prec=[]
val_prec=[]
train_rec=[]
val_rec=[]
train_f1=[]
val_f1=[]

best_val_loss=float('inf')


for epoch in range(epochs):
    model.train()
    total_loss, total_iou, total_acc, total_prec, total_rec, total_f1 = 0, 0, 0, 0, 0, 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.to(device)
            labels = labels.to(device)
        outputs = model(inputs)


        loss = criterion(labels, outputs)

        outputs_bin = (outputs > 0.5).float()  # Perform binarization on the GPU
        iou = iou_coef(labels, outputs_bin)
        acc = accuracy(labels, outputs_bin)
        prec = precision(labels, outputs_bin)
        rec = recall(labels, outputs_bin)
        f1 = f1_score(labels, outputs_bin)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou += iou.item()
        total_acc += acc.item()
        total_prec += prec.item()
        total_rec += rec.item()
        total_f1 += f1.item()

    avg_loss = total_loss / len(train_loader)
    avg_iou = total_iou / len(train_loader)
    avg_acc = total_acc / len(train_loader)
    avg_prec = total_prec / len(train_loader)
    avg_rec = total_rec / len(train_loader)
    avg_f1 = total_f1 / len(train_loader)

    train_losses.append(avg_loss)
    train_iou.append(avg_iou)
    train_acc.append(avg_acc)
    train_prec.append(avg_prec)
    train_rec.append(avg_rec)
    train_f1.append(avg_f1)

    print('Epoch [{}/{}], Loss: {:.4f}, IoU: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'
          .format(epoch+1, epochs, total_loss/ len(train_loader), total_iou / len(train_loader), total_acc / len(train_loader),
                  total_prec / len(train_loader), total_rec / len(train_loader), total_f1 / len(train_loader)))


    model.eval()
    total_val_loss = 0
    total_val_iou = 0
    total_val_acc = 0
    total_val_prec = 0
    total_val_rec = 0
    total_val_f1 = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.to(device)
                labels = labels.to(device)
            outputs = model(inputs)
            val_loss = criterion(labels, outputs)
            outputs_bin = (outputs > 0.5).float() 

            total_val_loss += val_loss.item()
            total_val_iou += iou_coef(labels, outputs_bin).item()
            total_val_acc += accuracy(labels, outputs_bin).item()
            total_val_prec += precision(labels, outputs_bin).item()
            total_val_rec += recall(labels, outputs_bin).item()
            total_val_f1 += f1_score(labels, outputs_bin).item()

        avg_val_loss = total_val_loss / len(test_loader)
        avg_val_iou = total_val_iou / len(test_loader)
        avg_val_acc = total_val_acc / len(test_loader)
        avg_val_prec = total_val_prec / len(test_loader)
        avg_val_rec = total_val_rec / len(test_loader)
        avg_val_f1 = total_val_f1 / len(test_loader)

        val_losses.append(avg_val_loss)
        val_iou.append(avg_val_iou)
        val_acc.append(avg_val_acc)
        val_prec.append(avg_val_prec)
        val_rec.append(avg_val_rec)
        val_f1.append(avg_val_f1)

    print(
        'Epoch [{}/{}], Validation Loss: {:.4f}, Val IoU: {:.4f}, Val Accuracy: {:.4f}, Val Precision: {:.4f}, Val Recall: {:.4f}, Val F1: {:.4f}'
        .format(epoch + 1, epochs, total_val_loss / len(test_loader), total_val_iou / len(test_loader), total_val_acc / len(test_loader),
                total_val_prec / len(test_loader), total_val_rec / len(test_loader), total_val_f1 / len(test_loader)))

    if total_val_loss / len(test_loader) < best_val_loss:
        best_val_loss = total_val_loss / len(test_loader)
        torch.save(model.module.state_dict() if num_gpus > 1 else model.state_dict(), 'model_weights.pth')

# plots
plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.close()

plt.figure()
plt.plot(train_iou, label='Training IoU')
plt.plot(val_iou, label='Validation IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.title('Training and Validation IoU')
plt.legend()
plt.savefig('iou_plot.png')
plt.close()

plt.figure()
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label = 'Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.close()

plt.figure()
plt.plot(train_prec, label='Training Precision')
plt.plot(val_prec, label='Validation Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Training and Validation Precision')
plt.legend()
plt.savefig('precision_plot.png')
plt.close()

plt.figure()
plt.plot(train_rec, label='Training Recall')
plt.plot(val_rec, label='Validation Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Training and Validation Recall')
plt.legend()
plt.savefig('recall_plot.png')
plt.close()

plt.figure()
plt.plot(train_f1, label='Training F1 Score')
plt.plot(val_f1, label='Validation F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('Training and Validation F1 Score')
plt.legend()
plt.savefig('f1_score_plot.png')
plt.close()


###########################Evaluation###################################################################################
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

model.eval()

true_labels = []
pred_labels = []

total_test_iou = 0
total_test_acc= 0
total_test_prec = 0
total_test_rec = 0
total_test_f1 = 0
total_test_loss = 0

for inputs, labels in test_loader:
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()
    outputs = model(inputs)
    outputs_bin = (outputs > 0.5).float()
    test_loss = criterion(labels, outputs_bin).item()
    total_test_loss += test_loss
    total_test_iou += iou_coef(labels, outputs_bin).item()
    total_test_acc += accuracy(labels, outputs_bin).item()
    total_test_prec += precision(labels, outputs_bin).item()
    total_test_rec += recall(labels, outputs_bin).item()
    total_test_f1 += f1_score(labels, outputs_bin).item()

    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    true_labels.extend(labels.reshape(-1))
    pred_labels.extend(outputs.reshape(-1))

print('Test Loss: {:.4f}, Test IoU: {:.4f}, Test Acc: {:.4f} Test Precision: {:.4f}, Test Recall: {:.4f}, Test F1: {:.4f}'
      .format(total_test_loss / len(test_loader), total_test_iou / len(test_loader), total_test_acc / len(test_loader),
              total_test_prec / len(test_loader), total_test_rec / len(test_loader), total_test_f1 / len(test_loader)))

fpr, tpr, _ = roc_curve(true_labels, pred_labels)
roc_auc = auc(fpr, tpr)

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



