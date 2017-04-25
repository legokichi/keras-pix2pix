from typing import Tuple, List, Text, Dict, Any, Iterator

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model



def create_unet(in_shape: Tuple[int,int,int], out_shape: Tuple[int,int,int], filters: int) -> Model:
    '''
    reference models
    * https://github.com/phillipi/pix2pix/blob/master/models.lua#L47
    * https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/pix2pix/src/model/models.py#L317
    '''
    input_tensor = Input(shape=in_shape) # type: Input
    output_ch = out_shape[2]
    # enc
    x =                       Conv2D(         filters*1, kernel_size=(4, 4), strides=(2, 2), padding="same")( input_tensor )       ; e1 = x
    x = BatchNormalization()( Conv2D(         filters*2, kernel_size=(4, 4), strides=(2, 2), padding="same")( LeakyReLU(0.2)(x) ) ); e2 = x
    x = BatchNormalization()( Conv2D(         filters*4, kernel_size=(4, 4), strides=(2, 2), padding="same")( LeakyReLU(0.2)(x) ) ); e3 = x
    x = BatchNormalization()( Conv2D(         filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same")( LeakyReLU(0.2)(x) ) ); e4 = x
    x = BatchNormalization()( Conv2D(         filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same")( LeakyReLU(0.2)(x) ) ); e5 = x
    x = BatchNormalization()( Conv2D(         filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same")( LeakyReLU(0.2)(x) ) ); e6 = x
    x = BatchNormalization()( Conv2D(         filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same")( LeakyReLU(0.2)(x) ) ); e7 = x
    x =                       Conv2D(         filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same")( LeakyReLU(0.2)(x) )  ; e8 = x
    # dec
    x = BatchNormalization()( Conv2DTranspose(filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same")( Activation("relu")(x) ) ); x = Concatenate()([Dropout(0.5)(x), e7])
    x = BatchNormalization()( Conv2DTranspose(filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same")( Activation("relu")(x) ) ); x = Concatenate()([Dropout(0.5)(x), e6])
    x = BatchNormalization()( Conv2DTranspose(filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same")( Activation("relu")(x) ) ); x = Concatenate()([Dropout(0.5)(x), e5])
    x = BatchNormalization()( Conv2DTranspose(filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same")( Activation("relu")(x) ) ); x = Concatenate()([x, e4])
    x = BatchNormalization()( Conv2DTranspose(filters*4, kernel_size=(4, 4), strides=(2, 2), padding="same")( Activation("relu")(x) ) ); x = Concatenate()([x, e3])
    x = BatchNormalization()( Conv2DTranspose(filters*2, kernel_size=(4, 4), strides=(2, 2), padding="same")( Activation("relu")(x) ) ); x = Concatenate()([x, e2])
    x = BatchNormalization()( Conv2DTranspose(filters*1, kernel_size=(4, 4), strides=(2, 2), padding="same")( Activation("relu")(x) ) ); x = Concatenate()([x, e1])
    x =                       Conv2DTranspose(output_ch, kernel_size=(4, 4), strides=(2, 2), padding="same")( Activation("relu")(x) )
    
    x = Activation("tanh")(x)
    
    unet = Model(inputs=[input_tensor], outputs=[x])
    
    return unet

def create_discriminator_patch(in_shape: Tuple[int,int,int], filters: int) -> Model:
    '''
    PatchGAN

    reference models
    * https://github.com/phillipi/pix2pix/blob/b479b6b7d37f9d7e87dce7f5e627dc3bb7b4a117/models.lua#L180
    * https://github.com/makora9143/pix2pix-keras-tensorflow/blob/4b7d2192607448659ba7b2c0b638d395dcd23ef4/model.py#L13
    * https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/pix2pix/src/model/models.py#L201

    patch_shape = (70, 70)
    if generator: (256, 256, 3) -> (256, 256, 3)
    then inshape == (70, 70, 3+3)
    '''

    input_tensor = Input(shape=in_shape) # type: Input

    x = LeakyReLU(0.2)(                       Conv2D(filters*1, kernel_size=(4, 4), strides=(2, 2), padding="same")( input_tensor ) )

    x = LeakyReLU(0.2)( BatchNormalization()( Conv2D(filters*2, kernel_size=(4, 4), strides=(2, 2), padding="same")(x) ) )
    x = LeakyReLU(0.2)( BatchNormalization()( Conv2D(filters*4, kernel_size=(4, 4), strides=(2, 2), padding="same")(x) ) )
    x = LeakyReLU(0.2)( BatchNormalization()( Conv2D(filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same")(x) ) )

    x = Activation("sigmoid")(                Conv2D(filters*1, kernel_size=(4, 4), strides=(2, 2), padding="same")(x) )
    
    x = Flatten()(x)#; x_flat = x  # to 1 dim array to use minibatch discrimination
    #x = Activation("softmax")( Dense(2)(x) )

    disc = Model(input=[input_tensor], output=[x])#, x_flat])

    return disc

def create_discriminator_image(in_shape: Tuple[int,int,int], filters: int):
    '''
    ImageGAN

    reference models
    * https://github.com/costapt/vess2ret/blob/master/models.py#L517
    '''
    return create_discriminator_patch(in_shape, filters)


def pix2pix():
    unet = create_unet((256, 256, 3), (256, 256, 3), 128)
    disc = create_discriminator_image((256, 256, 6), 64)

    a_real = Input(shape=(a_ch, 512, 512))
    b_real = Input(shape=(b_ch, 512, 512))
    b_fake = unet(a_real)

    disc(Concatenate()([a_real, b_real]))
    disc(Concatenate()([a_real, b_fake]))


if __name__ == '__main__':
    unet = create_unet((256, 256, 3), (256, 256, 3), 128)
    unet.summary()
    plot_model(unet, to_file='unet.png', show_shapes=True, show_layer_names=True)
    
    disc = create_discriminator_image((256, 256, 6), 64)
    disc.summary()
    plot_model(disc, to_file='disc.png', show_shapes=True, show_layer_names=True)
    
    exit()





def ___create_discriminator(in_shape: Tuple[int,int,int], filters: int):
    '''
    working on prog

    reference models
    * https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/pix2pix/src/model/models.py#L201
    '''

    input_tensors = [Input(shape=(70, 70, 3)) for i in range(nb_patch)]

    patches = [ create_discriminator_patch(input_tensor, filters) for input_tensor in input_tensors]

    xs = [x for (x, _) in patches]
    #x_flats = [x_flat for (_, x_flat) in patches]

    x = Activation("softmax")( Dense(2)( Concatenate()(xs) ) )
    #x_mbd = Activation("softmax")( Dense(2)( Concatenate()(x_flats) ) )

    # todo: minibatch discrimination - https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/pix2pix/src/model/models.py#L250

    discriminator = Model(inputs=input_tensors, outputs=[x])

    return discriminator


