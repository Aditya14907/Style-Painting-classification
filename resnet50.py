import tensorflow as tf
from keras import layers, models, Input
from keras import layers, models
import numpy as np
import os
import numpy as np
import itertools
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import time
import math

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
from keras.layers import BatchNormalization, Conv2D, Activation, Flatten, MaxPooling2D, Concatenate, Input, Softmax, Dense, GlobalAveragePooling2D,Lambda
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import itertools

def identity_block(input_tensor, filters, kernel_size):
    filters1, filters2, filters3 = filters

    x = layers.Conv2D(filters1, (1, 1), kernel_initializer='he_normal')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def conv_block(input_tensor, filters, kernel_size, strides=(2, 2)):
    filters1, filters2, filters3 = filters

    x = layers.Conv2D(filters1, (1, 1), strides=strides, kernel_initializer='he_normal')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal')(input_tensor)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def ResNet50(input_shape, num_classes):
    img_input = Input(shape=input_shape)

    x = layers.ZeroPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv_block(x, [64, 64, 256], (3, 3), strides=(1, 1))
    x = identity_block(x, [64, 64, 256], (3, 3))
    x = identity_block(x, [64, 64, 256], (3, 3))

    x = conv_block(x, [128, 128, 512], (3, 3))
    x = identity_block(x, [128, 128, 512], (3, 3))
    x = identity_block(x, [128, 128, 512], (3, 3))
    x = identity_block(x, [128, 128, 512], (3, 3))

    x = conv_block(x, [256, 256, 1024], (3, 3))
    x = identity_block(x, [256, 256, 1024], (3, 3))
    x = identity_block(x, [256, 256, 1024], (3, 3))
    x = identity_block(x, [256, 256, 1024], (3, 3))
    x = identity_block(x, [256, 256, 1024], (3, 3))
    x = identity_block(x, [256, 256, 1024], (3, 3))

    x = conv_block(x, [512, 512, 2048], (3, 3))
    x = identity_block(x, [512, 512, 2048], (3, 3))
    x = identity_block(x, [512, 512, 2048], (3, 3))

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(12, activation='softmax')(x)
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

LR_START = 1e-5
LR_MAX = 5e-3
LR_MIN = 1e-7
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
filepath="/home/nithin12113109/my_space/model_weights/"+ "resnet50.h5"
#Model Checkpoint
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
#Learning rate scheduler
lr_callback = LearningRateScheduler(lrfn,verbose=False)

input_shape = (224, 224, 3)
num_classes = 12  
model = ResNet50(input_shape, num_classes)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


start_time=time.time()
history=model.fit_generator(train_dataset, validation_data=valid_dataset, epochs=EPOCHS,callbacks=[checkpoint,lr_callback])

end_time=time.time()
tt=end_time-start_time
print("Total training time : ",tt)

y_pred = []
Y_pred =[]
# Iterate over batches in the validation dataset
for i in range(len(valid_dataset)):
        X, y = valid_dataset[i]
        preds = model.predict_on_batch(X)
        for pred in preds:
          Y_pred.append(pred)
          pred_label = np.argmax(pred)
          y_pred.append(pred_label)
y_pred = np.array(y_pred)
Y_pred = np.array(Y_pred)
y_true=np.array(valid_dataset.classes)

from sklearn import metrics
from sklearn.metrics import confusion_matrix
cm = metrics.confusion_matrix(y_true,y_pred)
print(cm)
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

id="/home/nithin12113109/my_space/Result/resnet50_result"
file = open(id+'.txt', 'w')
file.write("Total training time: "+str(tt)+" \n Accuracy Score: "+str(acc))
file.write("\nTop 1 Accuarcy :"+str(top1_acc))
file.write("\nTop 5 accuracy : "+str(top5_acc))

file.write("\n\nConfusion matrix: \n"+str(cm))
file.write("\n\n Precision : \n"+str(precision))
file.write("\n\n Recall score : \n"+str(recall))
file.write("\n\n F1 score : \n" +str(f1))
file.close()

np.savetxt(id+'_.csv', cm, delimiter=",")
