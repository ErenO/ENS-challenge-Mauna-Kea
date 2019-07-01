import os
import sys
import matplotlib.pyplot as plt
import cv2 as cv
import itertools
from skimage.transform import resize
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score

def plot_confusion_matrix(y, y_pred):
    plt.figure(figsize=(7, 6))
    plt.title('Confusion matrix', fontsize=16)
    plt.imshow(confusion_matrix(y, y_pred))
    plt.xticks(np.arange(4), classes, rotation=45, fontsize=12)
    plt.yticks(np.arange(4), classes, fontsize=12)
    plt.colorbar()
    plt.show()
    print("val accuracy:", accuracy_score(y, y_pred))

def read_and_resize(img):
    res = resize(img, (224, 224), preserve_range=True, mode='reflect')
    return np.expand_dims(res, 0)

def plot_history(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
def datagen_create(data_aug = False):
    if (data_aug):
#         datagen = ImageDataGenerator(
#         rescale = 1./255, 
# #         featurewise_center = False,  # set input mean to 0 over the dataset
# #         samplewise_center = False,  # set each sample mean to 0
# #         featurewise_std_normalization = False,  # divide inputs by std of the dataset
# #         samplewise_std_normalization = False,  # divide each input by its std
# #         zca_whitening = False,  # apply ZCA whitening
#         rotation_range = 90,  # randomly rotate images in the range (degrees, 0 to 180)
#         horizontal_flip = True,  # randomly flip images
#         vertical_flip = True)
#         return (datagen)
        datagen = ImageDataGenerator(
            rescale=1./255, 
            zoom_range = 0.3,
            width_shift_range = 0.3,
            height_shift_range=0.3,
            rotation_range = 90,  # randomly rotate images in the range (degrees, 0 to 180)
            horizontal_flip = True,  # randomly flip images
            vertical_flip = True
        )
        return (datagen)
    else:
        return ImageDataGenerator(rescale = 1./255)
    
def plotHistogram(a):
    """
    Plot histogram of RGB Pixel Intensities
    """
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(a)
    plt.axis('off')
    histo = plt.subplot(1,2,2)
    histo.set_ylabel('Count')
    histo.set_xlabel('Pixel Intensity')
    n_bins = 30
    plt.hist(a[:,:,0].flatten(), bins= n_bins, lw = 0, color='r', alpha=0.5);
    plt.hist(a[:,:,1].flatten(), bins= n_bins, lw = 0, color='g', alpha=0.5);
    plt.hist(a[:,:,2].flatten(), bins= n_bins, lw = 0, color='b', alpha=0.5);

    
#####################
### NEW FUNCTIONS ###
#####################

def plot2histo(lst, lst2, xStr, yStr, xStr2, yStr2):
    fig = plt.figure(figsize=(13, 13))

    ax = fig.add_subplot(2, 2, 1)
    ax.hist(lst, bins=10)
    ax.set_xlabel(xStr)
    ax.set_ylabel(yStr)

    ax = fig.add_subplot(2, 2, 2)
    ax.hist(lst2, bins=10)
    ax.set_xlabel(xStr2)
    ax.set_ylabel(yStr2)
    plt.show()
    
    
def plot_imgs(randomLst, classes, labeled_files, id_label_map, cols=8, rows=3):
    ind = 0
    fig = plt.figure(figsize=(3*cols-1, 3.5*rows-1))
    for i in range(cols):
        for j in range(rows):
            random_index = randomLst[ind]
            ax = fig.add_subplot(rows, cols, i * rows + j + 1)
            ax.grid('off')
            ax.axis('off')
            img = cv.imread(labeled_files[random_index])
            ax.imshow(img)
            imgName = labeled_files[random_index].split(os.path.sep)[-1]
            ax.set_title("{} {}\nShape: {}".format(classes[id_label_map[imgName]], id_label_map[imgName], img.shape))
            ind += 1
    plt.show()
    
def plot_imgs_by_id(imgNameLst, classes, labeled_files, id_label_map, cols=8, rows=3):
    ind = 0
    fig = plt.figure(figsize=(3*cols-1, 3.5*rows-1))
    for i in range(cols):
        for j in range(rows):
            imgName = imgNameLst[i][j]
            ax = fig.add_subplot(rows, cols, i * rows + j + 1)
            ax.grid('off')
            ax.axis('off')
            img = cv.imread(imgName)
            ax.imshow(img)
            imgName = imgName.split(os.path.sep)[-1]
            ax.set_title("{} {}\nShape: {}".format(classes[id_label_map[imgName]],
                                                   id_label_map[imgName],
                                                   img.shape))
            ind += 1
    plt.show()
    
def plot_pred_imgs(test_generator, classes, labeled_files, id_label_map, cols=8, rows=3):
    cols = 5
    rows = 2
    ind = 0
    #pour allouer un cadre
    randomLst = []
    for i in range(0, (cols * rows)):
        randomLst.append(np.random.randint(0, len(test_generator.filenames)))
    fig = plt.figure(figsize=(3 * cols - 1, 4 * rows - 1))
    for i in range(cols):
        for j in range(rows):
            random_index = randomLst[ind]
            ax = fig.add_subplot(rows, cols, i * rows + j + 1)
            ax.grid('off')
            ax.axis('off')
            imgName = test_generator.filepaths[random_index]
            img = cv.imread(imgName)
    #         print (np.argmax(model.predict(read_and_resize(img))[0]))
            ax.imshow(img)
            ax.set_title("Label: {}\nShape: {}\nPrediction: {}".format(id_label_map[imgName.split(os.path.sep)[-1]], img.shape, np.argmax(test_predictions[random_index])))
            ind += 1
    plt.show()