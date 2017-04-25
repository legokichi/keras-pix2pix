# ~~keras-pix2pix~~ keras-u-net

working in progress

ImageGAN only.

## ref

* https://github.com/phillipi/pix2pix
* https://github.com/affinelayer/pix2pix-tensorflow
* https://github.com/costapt/vess2ret
* https://github.com/makora9143/pix2pix-keras-tensorflow
* https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/pix2pix
* https://github.com/jocicmarko/ultrasound-nerve-segmentation/

## install

```
git clone --recursive https://github.com/legokichi/keras-pix2pix.git
pyenv shell anaconda3-4.1.1
sudo apt-get install graphviz
conda install theano pygpu
pip install tensorflow-gpu
pip install keras
pip install mypy
pip install pydot_ng
```


## type check

```
mypy --ignore-missing-imports train.py 
```

working in progress

## model

![segnet](https://raw.githubusercontent.com/legokichi/keras-pix2pix/master/unet.png)

![segnet](https://raw.githubusercontent.com/legokichi/keras-pix2pix/master/disc.png)
