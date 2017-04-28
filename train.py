from typing import Tuple, List, Text, Dict, Any, Iterator, Union, Sized, Callable, cast
from datetime import datetime
import argparse
import sys
sys.path.append("/usr/local/Cellar/opencv3/3.2.0/lib/python3.5/site-packages/") # mac opencv path
import cv2
import numpy as np
np.random.seed(2017) # for reproducibility
import os
#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ["THEANO_FLAGS"] = "exception_verbosity=high,optimizer=None,device=cpu"
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cpu,floatX=float32,optimizer=fast_compile'
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.backend import set_image_data_format
# keras.backend.backend()
# keras.backend.set_epsilon(1e-07)
# keras.backend.epsilon()
# keras.backend.set_floatx('float32')
# keras.backend.floatx()
# set_image_data_format('channels_first') # theano
set_image_data_format("channels_last")
# keras.backend.image_data_format()
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
from keras.optimizers import SGD, Adam
from keras.backend import tensorflow_backend
import keras.backend as K
from chainer.iterators import MultiprocessIterator, SerialIterator
from model_unet import create_unet
from mscoco import get_iter as get_coco

def convert_to_keras_batch(iter: Iterator[List[Tuple[np.ndarray, np.ndarray]]]) -> Iterator[Tuple[np.ndarray, np.ndarray]] :
    while True:
        batch = iter.__next__() # type: List[Tuple[np.ndarray, np.ndarray]]
        xs = [x for (x, _) in batch] # type: List[np.ndarray]
        ys = [y for (_, y) in batch] # type: List[np.ndarray]
        _xs = np.array(xs) # (n, 480, 360, 3)
        _ys = np.array(ys) # (n, 480, 360, n_classes)
        yield (_xs, _ys)

def dice_coef(y_true: K.tf.Tensor, y_pred: K.tf.Tensor) -> K.tf.Tensor:
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def dice_coef_loss(y_true: K.tf.Tensor, y_pred: K.tf.Tensor) -> K.tf.Tensor:
    return -dice_coef(y_true, y_pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='U-net trainer from mscoco')
    parser.add_argument("--epochs",  action='store', type=int, default=1000, help='epochs')
    parser.add_argument("--resume",  action='store', type=str, default="", help='*_weights.hdf5')
    parser.add_argument("--initial_epoch", action='store', type=int, default=0, help='initial_epoch')
    parser.add_argument("--ker_init", action='store', type=str, default="glorot_uniform", help='conv2D kernel initializer')
    parser.add_argument("--lr", action='store', type=float, default=0.001, help='learning late')
    parser.add_argument("--optimizer", action='store', type=str, default="adam", help='adam|nesterov')
    parser.add_argument("--filters", action='store', type=int, default=64, help='32|64|128')
    parser.add_argument("--dice_coef", action='store_true', help='use dice_coef for loss function')
    parser.add_argument("--dir", action='store', type=str, default="./", help='mscoco dir')
    args = parser.parse_args()

    name = args.dir
    name += datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name += "_fil" + str(args.filters)
    name += "_" + args.optimizer
    name += "_lr" + str(args.lr)
    name += "_" + args.ker_init
    if args.dice_coef: name += "_dice_coef"
    else: name += "_crossentropy"

    print("name: ", name)

    resize_shape = (512, 512)
    
    train, valid = get_coco(resize_shape, args.dice_coef, args.dir) # type: Tuple[Iterator[np.ndarray], Iterator[np.ndarray]]

    train_iter = convert_to_keras_batch(
        #SerialIterator(
        MultiprocessIterator(
            train,
            batch_size=8,
            n_processes=12,
            n_prefetch=120,
            shared_mem=1000*1000*5
        )
    ) # type: Iterator[Tuple[np.ndarray, np.ndarray]]

    valid_iter = convert_to_keras_batch(
        #SerialIterator(
        MultiprocessIterator(
            valid,
            batch_size=8,
            #repeat=False,
            shuffle=False,
            n_processes=12,
            n_prefetch=120,
            shared_mem=1000*1000*5
        )
    ) # type: Iterator[Tuple[np.ndarray, np.ndarray]]
    
    old_session = tensorflow_backend.get_session()

    with K.tf.Graph().as_default():
        session = K.tf.Session("")
        tensorflow_backend.set_session(session)
        tensorflow_backend.set_learning_phase(1)

        input_shape = (resize_shape[0], resize_shape[1], 3)
        if args.dice_coef:
            output_ch = 1
            loss = dice_coef_loss
            metrics = [dice_coef]
            filename = "_weights.epoch{epoch:04d}-val_loss{val_loss:.2f}-val_dice_coef{val_dice_coef:.2f}.hdf5"
        else:
            output_ch = 2
            loss = "categorical_crossentropy"
            metrics = ['accuracy']
            filename = "_weights.epoch{epoch:04d}-val_loss{val_loss:.2f}-val_acc{val_acc:.2f}.hdf5"

        model = create_unet(input_shape, output_ch, args.filters, args.ker_init)
        
        if args.optimizer == "nesterov":
            optimizer = SGD(lr=args.lr, momentum=0.9, decay=0.0005, nesterov=True)
        else:
            optimizer = Adam(lr=args.lr)#, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)

        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        if len(args.resume) > 0:
            model.load_weights(args.resume)

        with open(name+'_model.json', 'w') as f: f.write(model.to_json())

        callbacks = [] # type: List[Callback]

        callbacks.append(ModelCheckpoint(
            name+filename,
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
            period=1,
        ))

        callbacks.append(TensorBoard(
            log_dir=name+'_log',
            histogram_freq=1,
            write_graph=False,
            write_images=False,
        ))

        hist = model.fit_generator(
            generator=train_iter,
            steps_per_epoch=int(len(cast(Sized, train))/8),
            epochs=args.epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=valid_iter,
            validation_steps=30,
            initial_epoch=args.initial_epoch,
        )

        model.save_weights(name+'_weight_final.hdf5')
        with open(name+'_history.json', 'w') as f: f.write(repr(hist.history))
        
    tensorflow_backend.set_session(old_session)

    
