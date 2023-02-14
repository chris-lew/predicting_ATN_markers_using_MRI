import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import math
from scipy.ndimage import rotate
import numpy as np
from sklearn.metrics import auc, roc_curve, average_precision_score, precision_recall_curve, PrecisionRecallDisplay, roc_auc_score

def data_status(arr):
    # print data status: data shape, data range, data type.
    print('''-----------------------
    Data was loaded.
    Dataset shape: {}
    Data range: ({}, {})
    Data type: {}'''.format(
        arr.shape, 
        arr.min(), 
        arr.max(), 
        arr.dtype))
    
# Python program to count the frequency of elements in a list using a dictionary
  
def count_frequency(my_list):
  
    # Creating an empty dictionary 
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
  
    for key, value in freq.items():
        print ("% d : % d"%(key, value))
        
def training_metrics(history, accuracy_str):

    epochs_range = range(len(history['loss']))
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs_range, history['loss'], label='loss', color='C0')
    plt.plot(epochs_range, history['val_loss'], label='val_loss', color='C1')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')

    plt.subplot(1,2,2)
    plt.plot(epochs_range, history[accuracy_str], label=accuracy_str, color='C0')
    plt.plot(epochs_range, history['val_'+accuracy_str], label='val_'+accuracy_str, color='C1')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics ({})'.format(accuracy_str))
    plt.legend(loc='best')

    plt.show()
    
    print('Final Epoch Metrics')
    print('Training Loss: {}'.format(history['loss'][-1]))
    print('Validation Loss: {}'.format(history['val_loss'][-1]))
    print('Training {}: {}'.format(accuracy_str, history[accuracy_str][-1]))
    print('Validation {}: {}'.format(accuracy_str, history['val_'+accuracy_str][-1]))
    print()
    if accuracy_str == 'binary_accuracy':
        print('Best Training {}: {}'.format(accuracy_str, max(history[accuracy_str])))
        print('Best Validation {}: {}'.format(accuracy_str, max(history['val_'+accuracy_str])))
    else:
        print('Best Training {}: {}'.format(accuracy_str, min(history[accuracy_str])))
        print('Best Validation {}: {}'.format(accuracy_str, min(history['val_'+accuracy_str])))

    
def training_metrics_np_object(history, accuracy_str):

    epochs_range = range(len(history.item()['loss']))
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs_range, history.item()['loss'], label='loss', color='C0')
    plt.plot(epochs_range, history.item()['val_loss'], label='val_loss', color='C1')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')

    plt.subplot(1,2,2)
    plt.plot(epochs_range, history.item()[accuracy_str], label=accuracy_str, color='C0')
    plt.plot(epochs_range, history.item()['val_'+accuracy_str], label='val_'+accuracy_str, color='C1')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics ({})'.format(accuracy_str))
    plt.legend(loc='best')

    plt.show()
    
    print('Final Epoch Metrics')
    print('Training Loss: {}'.format(history.item()['loss'][-1]))
    print('Validation Loss: {}'.format(history.item()['val_loss'][-1]))
    print('Training {}: {}'.format(accuracy_str, history.item()[accuracy_str][-1]))
    print('Validation {}: {}'.format(accuracy_str, history.item()['val_'+accuracy_str][-1]))
    print()
    if accuracy_str == 'binary_accuracy':
        print('Best Training {}: {}'.format(accuracy_str, max(history.item()[accuracy_str])))
        print('Best Validation {}: {}'.format(accuracy_str, max(history.item()['val_'+accuracy_str])))
    else:
        print('Best Training {}: {}'.format(accuracy_str, min(history.item()[accuracy_str])))
        print('Best Validation {}: {}'.format(accuracy_str, min(history.item()['val_'+accuracy_str])))
    
def model_scores(y_true, y_preds):
    plt.figure(figsize=(12,5))
    
    try:
    
        fpr, tpr, _ = roc_curve(y_true, y_preds)
        roc_auc = auc(fpr, tpr)

        plt.subplot(1,2,1)
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")


        precision, recall, thresholds = precision_recall_curve(y_true, y_preds)
        ap_score = average_precision_score(y_true, y_preds)
        plt.subplot(1,2,2)
        plt.plot(recall, precision, color='darkorange',
                 lw=lw, label='Average Precision Score: %0.2f' % ap_score)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision (PPV)')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower right")

        plt.show()
        
    except:
        print('AUROC:', roc_auc_score(y_true, y_preds))
    
def display_image_with_pred_label(image_npy, predictions, labels, zscored=False):
    count = labels.shape[0]
    n_rows = int(math.ceil(count/5))
    plt.figure(figsize=(15, n_rows*3))
    
    for i in range(count):
        plt.subplot(n_rows, 5, i+1)
        rotated_img = rotate(image_npy[i, :, :, 1], 90)
        if zscored:
            plt.imshow(rotated_img, cmap='gray', vmin=-12, vmax=12)
        else:
            plt.imshow(rotated_img, cmap='gray')
        plt.axis('off')
        
        guess = 1 if predictions[i] > 0.5 else 0
        
        pos_neg = '[+]' if guess == labels[i] else '[-]'
        
        plt.title('{} #{:04d}:{:.3f}/{}'.format(pos_neg, i, float(predictions[i]), float(labels[i])))

    plt.show()
    
    
