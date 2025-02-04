'''Train a simple deep CNN on the CIFAR10 small images dataset.
GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py
It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from sklearn import cross_validation
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

batch_size = 16
nb_classes = 10
nb_epoch = 4
not_data_augmentation = True

perform_cv = False

img_rows, img_cols = 32, 32
img_channels = 3


(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

def create_model():
    model = Sequential()
    
    model.add(Convolution2D(64, 3, 3,border_mode='same',input_shape=X_train.shape[1:], subsample=(1, 1)))
    model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
    
    model.add(Convolution2D(64, 3, 3,border_mode='same',subsample=(1, 1)))
    model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
    
    model.add(Convolution2D(64, 3, 3,border_mode='same',subsample=(1, 1)))
    model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
    
    model.add(MaxPooling2D(pool_size = (2,2), strides = (1,1)))
    
    model.add(Convolution2D(128, 3, 3,border_mode='same',subsample=(1, 1)))
    model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
    
    model.add(Convolution2D(128, 3, 3,border_mode='same',subsample=(1, 1)))
    model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
    
    model.add(Convolution2D(128, 3, 3,border_mode='same',subsample=(1, 1)))
    model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
    
    model.add(MaxPooling2D(pool_size = (2,2), strides = (1,1))) 
    
    model.add(Flatten())
    model.add(Dense(256))
    model.add(keras.layers.core.Dropout(0.5))
    model.add(Dense(256))
    model.add(keras.layers.core.Dropout(0.5))
    model.add(Dense(nb_classes, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
    return model

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
model = create_model()
if not_data_augmentation:
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)
'''
else:
    print('with data augmentation')
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA
        rotation_range=0,  # randomly rotate images in the range
        width_shift_range=0.1,  # randomly shift images horizontally
        height_shift_range=0.1,  # randomly shift images vertically
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test))
'''
                        
def evaluate_cv(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)
            
if perform_cv:
    print('performing kfold cross validation')
    k_folds = 3
    skf = cross_validation.StratifiedKfold(y_test, n_folds = k_folds, shuffle=True)
    for i, (train, test) in enumerate(sfk):
        model = None
        model = create_model()
        evaluate_cv(model, X_train[train], Y_train[train], X_test[test], Y_test[test])