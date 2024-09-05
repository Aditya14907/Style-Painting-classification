import tensorflow as tf
from keras import layers,models,regularizers
from keras.preprocessing.image import ImageDataGenerator 
from keras.layers import BatchNormalization, Conv2D, Activation, Flatten, MaxPooling2D, Concatenate, Input, Softmax, Dense, GlobalAveragePooling2D,ZeroPadding2D,Dropout
from keras.callbacks import Callback
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

import itertools

import warnings
from functools import partial
from numbers import Integral, Real


from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import label_binarize
from sklearn.utils import (
    assert_all_finite,
    check_array,
    check_consistent_length,
    column_or_1d,
)
from sklearn.utils.multiclass import type_of_target

def dense_block(x,blocks,name):
    for i in range(blocks):
        x=conv_block(x,48,name=name+'_block'+str(i+1))
    return x
  
def conv_block(x,growth_rate,name):
    bn_axis=3
    x1 = BatchNormalization(axis=bn_axis,epsilon=1.001e-5,name=name+'_0_bn')(x)
    x1 = Activation('relu',name=name+'_0_relu')(x1)
    x1 = Conv2D(  4*growth_rate,1,use_bias=False,name=name+'_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3,padding='same',use_bias=False,name=name + '_2_conv')(x1)
    x  = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x 
  
def transition_block(x, reduction, name):
    bn_axis = 3
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(int(x.shape[bn_axis]) * reduction), 1,use_bias=False,name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x    

def DenseNet(blocks,input_shape,num_classes=12):
    bn_axis = 3

    img_input = Input(shape=input_shape)
    
    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = Conv2D(96, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = Activation('relu', name='relu')(x)

    x = GlobalAveragePooling2D(name='avg_pool')(x) 
    x = Dense(256,activation='relu',name='fc')(x)
    x = Dropout(0.35)(x)
    x = Dense(num_classes,activation='softmax',name='fc1')(x)
    

    inputs = img_input
    model = models.Model(inputs, x, name='densenet161')

    return model   
    
def load_datasets(train, test, image_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    
    train_generator = train_datagen.flow_from_directory(
        train,
        target_size=image_size,
        class_mode='categorical',
        batch_size=batch_size,
        classes=sorted(os.listdir(train))
    )
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        test,
        target_size=image_size,
        class_mode='categorical',
        batch_size=batch_size,
        classes=sorted(os.listdir(test))
    )
    return train_generator, validation_generator

LR_START = 1e-06
LR_MAX = 5e-5
LR_MIN = 1e-9
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 15
EPOCHS = 50
def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        decay_total_epochs = EPOCHS - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS - 1
        decay_epoch_index = epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        phase = math.pi * decay_epoch_index / decay_total_epochs
        cosine_decay = 0.5 * (1 + math.cos(phase))
        lr = (LR_MAX - LR_MIN) * cosine_decay + LR_MIN
    return lr
  
train_dataset, valid_dataset = load_datasets('dataset/images/training', 'dataset/images/test', image_size=(224, 224), batch_size=32)

blocks = [6, 12, 36, 24]  
model = DenseNet(blocks=blocks,input_shape=(224, 224, 3),num_classes=12)

filepath="/home/nithin12113109/my_space/model_weights/"+ "densenet161.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
lr_callback = LearningRateScheduler(lrfn,verbose=False)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()   

start_time=time.time()
history=model.fit_generator(train_dataset, validation_data=valid_dataset, epochs=EPOCHS,callbacks=[checkpoint,lr_callback])
end_time=time.time()
tt=end_time-start_time

val_loss, val_acc = model.evaluate_generator(valid_dataset, verbose=0)

y_pred = []
Y_pred =[]
for i in range(len(valid_dataset)):
        X, y = valid_dataset[i]
        preds = model.predict_on_batch(X)
        for pred in preds:
          Y_pred.append(pred)
          pred_label = np.argmax(pred)
          y_pred.append(pred_label)
y_pred = np.array(y_pred)
Y_pred = np.array(Y_pred)

from sklearn import metrics
from sklearn.metrics import confusion_matrix
cm = metrics.confusion_matrix(y_true,y_pred)

from sklearn.metrics import precision_score
precision = precision_score(y_true, y_pred, average='micro')

from sklearn.metrics import recall_score
recall = recall_score(y_true, y_pred, average='micro')

from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred, average='micro')


def top_k_accuracy_score(y_true, y_score, k, normalize=True, sample_weight=None, labels=None):
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_true = column_or_1d(y_true)
    y_type = type_of_target(y_true)
    if y_type == "binary" and labels is not None and len(labels) > 2:
        y_type = "multiclass"
    if y_type not in {"binary", "multiclass"}:
        raise ValueError(
            "y type must be 'binary' or 'multiclass', got '{}' instead.".format(y_type)
        )
    y_score = check_array(y_score, ensure_2d=False)
    if y_type == "binary":
        if y_score.ndim == 2 and y_score.shape[1] != 1:
            raise ValueError(
                "'y_true' is binary while y_score is 2d with"
                " {} classes. If 'y_true' does not contain all the"
                "labels, 'labels' must be provided".format(y_score.shape[1])
            )
        y_score = column_or_1d(y_score)

    check_consistent_length(y_true, y_score, sample_weight)
    y_score_n_classes = y_score.shape[1] if y_score.ndim == 2 else 2

    if labels is None:
        classes = np.unique(y_true)
        n_classes = len(classes)

        if n_classes != y_score_n_classes:
            raise ValueError(
                "Number of classes in 'y_true' ({}) not equal "
                "to the number of classes in 'y_score' ({})."
                "You can provide a list of all known classes by assigning it "
                "to the 'labels' parameter.".format(n_classes,y_score_n_classes)
            )
    else:
        labels = column_or_1d(labels)
        classes = np.unique(labels)
        n_labels = len(labels)
        n_classes = len(classes)

        if n_classes != n_labels:
            raise ValueError("Parameter 'labels' must be unique.")

        if not np.array_equal(classes, labels):
            raise ValueError("Parameter 'labels' must be ordered.")

        if n_classes != y_score_n_classes:
            raise ValueError(
                "Number of given labels ({}) not equal to the "
                "number of classes in 'y_score' ({}).".format(n_classes,y_score_n_classes)
            )

        if len(np.setdiff1d(y_true, classes)):
            raise ValueError("'y_true' contains labels not in parameter 'labels'.")

    y_true_encoded = label_binarize(y_true, classes=classes)
    if y_type == "binary":
        if k == 1:
            threshold = 0.5 if y_score.min() >= 0 and y_score.max() <= 1 else 0
            y_pred = (y_score > threshold).astype(np.int64)
            hits = y_pred == y_true_encoded
        else:
            hits = np.ones_like(y_score, dtype=np.bool_)
    elif y_type == "multiclass":
        sorted_pred = np.argsort(y_score, axis=1, kind="mergesort")[:, ::-1]
        hits = np.zeros(len(y_true), dtype=bool)
        
        for i in range(len(y_true)):
            hits[i] = y_true[i] in sorted_pred[i, :k]
          
    if normalize:
        return np.average(hits, weights=sample_weight)
    elif sample_weight is None:
        return np.sum(hits)
    else:
        return np.dot(hits, sample_weight)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_true, y_pred)

top1_acc = top_k_accuracy_score(y_true,Y_pred,k=1)
top5_acc = top_k_accuracy_score(y_true,Y_pred,k=5)

id="/home/nithin12113109/my_space/Result/densenet161/result"
file = open(id+'.txt', 'w')
file.write("Total training time: "+str(tt)+" \n Accuracy Score: "+str(acc))
file.write("\nTop 1 Accuarcy :"+str(top1_acc))
file.write("\nTop 5 accuracy : "+str(top5_acc))
file.write("Validation accuracy : "+str(val_acc))
file.write("\n\nConfusion matrix: \n"+str(cm))
file.write("\n\n Precision : \n"+str(precision))
file.write("\n\n Recall score : \n"+str(recall))
file.write("\n\n F1 score : \n" +str(f1))
file.close()

np.savetxt(id+'_.csv', cm, delimiter=",")

history_a = history.history
loss = history_a['loss']
val_loss = history_a['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label="Training Loss")
plt.plot(epochs, val_loss, 'r', label="Validation Loss")
plt.title("Training Vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

accuracy = history_a['acc']
val_accuracy = history_a['val_acc']
plt.plot(epochs, accuracy, 'y', label="Training Accuracy")
plt.plot(epochs, val_accuracy, 'r', label="Validation Accuracy")
plt.title("Training Vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
