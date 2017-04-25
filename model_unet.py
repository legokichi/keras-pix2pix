from typing import Tuple, List, Text, Dict, Any, Iterator

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model



def create_unet(in_shape: Tuple[int,int,int], output_ch: int, filters: int, ker_init: str="glorot_uniform") -> Model:
    '''
    reference models
    * https://github.com/phillipi/pix2pix/blob/master/models.lua#L47
    * https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/pix2pix/src/model/models.py#L317
    '''
    input_tensor = Input(shape=in_shape) # type: Input
    # enc
    x =                       Conv2D(         filters*1, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( input_tensor )       ; e1 = x
    x = BatchNormalization()( Conv2D(         filters*2, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( LeakyReLU(0.2)(x) ) ); e2 = x
    x = BatchNormalization()( Conv2D(         filters*4, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( LeakyReLU(0.2)(x) ) ); e3 = x
    x = BatchNormalization()( Conv2D(         filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( LeakyReLU(0.2)(x) ) ); e4 = x
    x = BatchNormalization()( Conv2D(         filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( LeakyReLU(0.2)(x) ) ); e5 = x
    x = BatchNormalization()( Conv2D(         filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( LeakyReLU(0.2)(x) ) ); e6 = x
    x = BatchNormalization()( Conv2D(         filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( LeakyReLU(0.2)(x) ) ); e7 = x
    x =                       Conv2D(         filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( LeakyReLU(0.2)(x) )  ; e8 = x
    # dec
    x = BatchNormalization()( Conv2DTranspose(filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) ) ); x = Concatenate(axis=3)([Dropout(0.5)(x), e7])
    x = BatchNormalization()( Conv2DTranspose(filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) ) ); x = Concatenate(axis=3)([Dropout(0.5)(x), e6])
    x = BatchNormalization()( Conv2DTranspose(filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) ) ); x = Concatenate(axis=3)([Dropout(0.5)(x), e5])
    x = BatchNormalization()( Conv2DTranspose(filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) ) ); x = Concatenate(axis=3)([x, e4])
    x = BatchNormalization()( Conv2DTranspose(filters*4, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) ) ); x = Concatenate(axis=3)([x, e3])
    x = BatchNormalization()( Conv2DTranspose(filters*2, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) ) ); x = Concatenate(axis=3)([x, e2])
    x = BatchNormalization()( Conv2DTranspose(filters*1, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) ) ); x = Concatenate(axis=3)([x, e1])
    x =                       Conv2DTranspose(output_ch, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=ker_init)( Activation("relu")(x) )
    
    x = Activation("tanh")(x)
    #x = Activation('sigmoid')(x)
    
    unet = Model(inputs=[input_tensor], outputs=[x])
    
    return unet


if __name__ == '__main__':
    unet = create_unet((512, 512, 3), 1, 64, "he_normal")
    unet.summary()
    plot_model(unet, to_file='unet.png', show_shapes=True, show_layer_names=True)
    
    exit()
