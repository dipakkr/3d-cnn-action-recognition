import argparse
import os

import matplotlib
matplotlib.use('AGG')

import matplotlib.pyplot as plt
import numpy as np

from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, MaxPooling2D)

from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split

import videoto3d
from tqdm import tqdm

from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Dense
import keras
import math


##
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
    


def loaddata(video_dir, vid3d, nclass, result_dir, color=False, skip=True):
    
    files = os.listdir(video_dir)    
    X = []
    labels = []
    labellist = []

    pbar = tqdm(total=len(files))

    # 101 classes
    for filename in files:
        pbar.update(1)
        if filename == '.DS_Store':
            continue
        name = os.path.join(video_dir, filename)
        
        for v_files in os.listdir(name):
            
            v_file_path = os.path.join(name, v_files)
            label = vid3d.get_UCF_classname(filename)
            if label not in labellist:
                if len(labellist) >= nclass:
                    continue
                labellist.append(label)
            labels.append(label)
            X.append(vid3d.video3d(v_file_path, color=color, skip=skip))
           

    pbar.close()

    with open(os.path.join(result_dir, 'classes.txt'), 'w') as fp:
        for i in range(len(labellist)):
            fp.write('{}\n'.format(labellist[i]))

    for num, label in enumerate(labellist):
        for i in range(len(labels)):
            if label == labels[i]:
                labels[i] = num
    
    if color:
        return np.array(X).transpose((0, 2, 3, 4, 1)), labels
    else:
        return np.array(X).transpose((0, 2, 3, 1)), labels


def myGenerator(X_train, X_test, Y_train, Y_test,nb_classes, nb_batch):

    img_rows = 224
    img_cols = 224
    
    # Y_train = np_utils.to_categorical(Y_train, nb_classes)

    
    # X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 10, 3)
    
    # X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 10, 3)
    
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')

    # X_train /= 255
    # X_test /= 255

    t = X_train.shape[0]/nb_batch

    steps = math.floor(t)

    while 1:
        for i in range(steps): # 1875 * 32 = 60000 -> # of training samples            
            if i%125==0:
                print(i)
            yield X_train[i*nb_batch:(i+1)*nb_batch], Y_train[i*nb_batch:(i+1)*nb_batch]



def main():
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--videos', type=str, default='UCF101',
                        help='directory where videos are stored')
    parser.add_argument('--nclass', type=int, default=101)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--color', type=bool, default=False)
    parser.add_argument('--skip', type=bool, default=True)
    parser.add_argument('--depth', type=int, default=10)
    args = parser.parse_args()

    img_rows, img_cols  = 224, 224

    frames  = args.depth
    nb_classes = args.nclass
    nb_batch = args.batch

    channel = 3 if args.color else 1

    fname_npz = 'dataset_{}_{}_{}.npz'.format(
        args.nclass, args.depth, args.skip)

    vid3d = videoto3d.Videoto3D(img_rows, img_cols, frames)
     
    if os.path.exists(fname_npz):
        loadeddata = np.load(fname_npz)
        X, Y = loadeddata["X"], loadeddata["Y"]
    else:
        x, y = loaddata(args.videos, vid3d, nb_classes,
                        args.output, args.color, args.skip)

        print(x.shape)
        print(x.size)

        X = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
        
        Y = np_utils.to_categorical(y, nb_classes)

        X = X.astype('float32')

        np.savez(fname_npz, X=X, Y=Y)
        
        print('Saved dataset to dataset.npz.')
    
    print('X_shape:{}\nY_shape:{}'.format(X.shape, Y.shape))

    # Define model
    input_x = Input(shape = (img_rows, img_cols, frames, channel))

    initial_conv = Conv3D(16, kernel_size= (3, 3, 3), padding='same')(input_x)
    initial_conv = LeakyReLU(alpha=.001)(initial_conv)

    initial_conv = Conv3D(32, kernel_size= (3, 3, 3), padding='same')(initial_conv)
    initial_conv = LeakyReLU(alpha=.001)(initial_conv)

    ###########################
    # PARALLEL 1

    conv1 = Conv3D(16, kernel_size=(1, 1, 1),padding='same')(initial_conv)
    conv1 = LeakyReLU(alpha=.001)(conv1)
    conv1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv1)

    conv1 = Conv3D(16, kernel_size=(3, 3, 3),padding='same')(conv1)
    conv1 = LeakyReLU(alpha=.001)(conv1)
    
    conv1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv1)
    conv1 = Conv3D(1, kernel_size=(1, 1, 1),padding='same')(conv1)
    conv1 = LeakyReLU(alpha=.001)(conv1)
    ##############################

    #Parallel 2

    conv2 = Conv3D(8, kernel_size=(1, 1, 1),padding='same')(initial_conv)
    conv2 = LeakyReLU(alpha=.001)(conv2)

    conv2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)
    conv2 = Conv3D(8, kernel_size=(3, 3, 3),padding='same')(conv2)
    conv2 = LeakyReLU(alpha=.001)(conv2)
    
    conv2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)
    conv2 = Conv3D(1, kernel_size=(1, 1, 1),padding='same')(conv2)
    conv2 = LeakyReLU(alpha=.001)(conv2)
    ##############################

    #Parallel 3

    conv3 = Conv3D(4, kernel_size=(1, 1, 1),padding='same')(initial_conv)
    conv3 = LeakyReLU(alpha=.001)(conv3)
    conv3 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv3)

    conv3 = Conv3D(4, kernel_size=(3, 3, 3),padding='same')(conv3)
    conv3 = LeakyReLU(alpha=.001)(conv3)

    conv3 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv3)
    conv3 = Conv3D(1, kernel_size=(1, 1, 1),padding='same')(conv3)
    conv3 = LeakyReLU(alpha=.001)(conv3)
    ###################################

    added = keras.layers.Add()([conv1, conv2, conv3])
    added = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(added)
    
    added = Flatten()(added)

    dense_1 = Dense(784)(added)
    dense_2 = Dense(nb_classes)(dense_1)

    print(dense_2.shape)

    model = Model(input_x, dense_2)
    
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(), metrics=['accuracy']) 
    
    model.summary() 

    # plot_model(model, show_shapes=True,
    #            to_file=os.path.join(args.output, 'model.png'))

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=43)

    print(X_train.shape)
    print(X_test.shape)

    ####################

    # MODEL CHECK POINTS

    filepath="saved_models/dk_3dcnnmodel-{epoch:02d}-{val_acc:.2f}.hd5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

     # GPU CONFIGURATION 
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))

    print(X_train.shape)
    print(Y_train.shape)
    
    # history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=args.batch,
    #                     epochs=args.epoch, verbose=1, shuffle=True, callbacks=callbacks_list)

    history = model.fit_generator(myGenerator(X_train, X_test, Y_train, Y_test, nb_classes, nb_batch), samples_per_epoch = X_train.shape[0], epochs = args.epoch, verbose=1, callbacks=callbacks_list, shuffle = True)

    
    model.evaluate(X_test, Y_test, verbose=0)
    
    model_json = model.to_json()
    
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    with open(os.path.join(args.output, 'ucf101_3dcnnmodel.json'), 'w') as json_file:
        json_file.write(model_json)
    
    model.save_weights(os.path.join(args.output, 'ucf101_3dcnnmodel-gpu.hd5'))

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    
    print('Test loss:', loss)
    print('Test accuracy:', acc)

    plot_history(history, args.output)
    save_history(history, args.output)


if __name__ == '__main__':
    main()