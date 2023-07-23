# Define the metrics
def dice_loss(y_true, y_pred, smooth=1):
    intersection = (y_true * y_pred).sum()
    return 1- (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)

def iou_coef(y_true, y_pred, smooth=1):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def precision(y_true, y_pred):
    true_positives = (y_true * y_pred).sum()
    predicted_positives = y_pred.sum()
    precision = true_positives / (predicted_positives + 1e-7)
    return precision

def recall(y_true, y_pred):
    true_positives = (y_true * y_pred).sum()
    possible_positives = y_true.sum()
    recall = true_positives / (possible_positives + 1e-7)
    return recall

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) + 1e-7) / (p + r + 1e-7)

def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-5):
    y_true_pos = y_true.view(-1)
    y_pred_pos = y_pred.view(-1)
    true_pos = (y_true_pos * y_pred_pos).sum()
    false_neg = (y_true_pos * (1-y_pred_pos)).sum()
    false_pos = ((1-y_true_pos)*y_pred_pos).sum()
    return 1 - (true_pos + smooth) / (true_pos + alpha*false_neg + beta*false_pos + smooth)