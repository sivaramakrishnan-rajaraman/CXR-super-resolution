#super-resolution code
#%% load libraries

import tensorflow as tf
tf.keras.backend.clear_session()

#%%
import time
import numpy as np
from tqdm import tqdm
import math
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras import backend as K
from keras.losses import mean_absolute_error
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, BatchNormalization, Input, add, Activation
import cv2
from PIL import Image
from skimage.measure import compare_ssim
from matplotlib import pyplot as plt
from keras.preprocessing import image
import imageio

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" #if using multiple gpus, otherwise comment

#%% function to load train and test images

def to_dirname(name):
    if name[-1:] == '/':
        return name
    else:
        return name + '/'
    
def show(image):
    image = image[0] * 255
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image.show()

def save_model(model, name):
    json = model.to_json()
    with open(name, 'w') as f:
        f.write(json)

def load_model(name):
    with open(name) as f:
        json = f.read()
    model = model_from_json(json)
    return model
    
def load_images_test(name, size):
    x_images = []
    y_images = []
    for file in tqdm(os.listdir(name)):
        image = Image.open(name+file)
        if image.mode != "RGB":
            image.convert("RGB")
        x_image = image.resize((size[0]//2, size[1]//2)) #change to 3, 4, or 8 for varying scale factors
        x_image = x_image.resize(size, Image.BICUBIC) 
        x_image = np.array(x_image)
        y_image = image.resize(size)
        y_image = np.array(y_image)
        x_images.append(x_image)
        y_images.append(y_image)
    x_images = np.array(x_images)
    y_images = np.array(y_images)
    x_images = x_images / 255
    y_images = y_images / 255
    x_images = x_images.reshape(x_images.shape[0], size[0]//1, size[1]//1, 3) #for grayscale, change the number of channels to 1
    y_images = y_images.reshape(y_images.shape[0], size[0]//1, size[1]//1, 3) #for grayscale, change the number of channels to 1
    return x_images, y_images

def load_images_train(name, size):
    x_images = []
    y_images = []
    for file in tqdm(os.listdir(name)):
        image = Image.open(name+file)
        if image.mode != "RGB": # comment these lines if using grasycale images
            image.convert("RGB") # comment these lines if using grasycale images
        #train for multiple scale factors
        x_image1 = image.resize((size[0]//2, size[1]//2))
        x_image1 = x_image1.resize(size, Image.BICUBIC) 
        x_image2 = image.resize((size[0]//3, size[1]//3)) 
        x_image2= x_image2.resize(size, Image.BICUBIC) 
        x_image3 = image.resize((size[0]//4, size[1]//4)) 
        x_image3= x_image3.resize(size, Image.BICUBIC) 
        x_image4 = image.resize((size[0]//8, size[1]//8)) 
        x_image4= x_image4.resize(size, Image.BICUBIC) 
        
        x_image1 = np.array(x_image1)
        y_image1 = image.resize(size)
        y_image1 = np.array(y_image1)
        x_images.append(x_image1)
        y_images.append(y_image1)
        
        x_image2 = np.array(x_image2)
        y_image2 = image.resize(size)
        y_image2 = np.array(y_image2)
        x_images.append(x_image2)
        y_images.append(y_image2)
        
        x_image3 = np.array(x_image3)
        y_image3 = image.resize(size)
        y_image3 = np.array(y_image3)
        x_images.append(x_image3)
        y_images.append(y_image3)
        
        x_image4 = np.array(x_image4)
        y_image4 = image.resize(size)
        y_image4 = np.array(y_image4)
        x_images.append(x_image4)
        y_images.append(y_image4)      
        
    x_images = np.array(x_images)
    y_images = np.array(y_images)
    x_images = x_images / 255
    y_images = y_images / 255
    x_images = x_images.reshape(x_images.shape[0], size[0]//1, size[1]//1, 3) #for grayscale, change the number of channels to 1
    y_images = y_images.reshape(y_images.shape[0], size[0]//1, size[1]//1, 3) #for grayscale, change the number of channels to 1
    return x_images, y_images

#%% #define loss functions

def tf_log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def PSNR(y_true, y_pred):
	max_pixel = 1.0
	return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true)))) 

def ssim(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def ssim_loss(y_true, y_pred):
  return 1-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def ssim_multi(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, 1.0))

def ssim_multi_loss(y_true, y_pred):
    return 1-tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, 1.0))

def loss_mix(y_true, y_pred):
    return 0.6 * mean_absolute_error(y_true, y_pred) + \
            0.4 * (1-ssim(y_true, y_pred))

def loss_mix_multi(y_true, y_pred):
    return 0.6 * mean_absolute_error(y_true, y_pred) + \
            0.4 * (1-ssim_multi(y_true, y_pred)) 

#%% for multi gpu training and avoid issues due to model check point
#from https://github.com/keras-team/keras/issues/2436#issuecomment-354882296

class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

#%% define model: vdsr architecture
        
def vdsr(input_size=(256,256,3)):    #modify to 1 channel if using grayscale images
    model_input = Input(shape=(256,256,3)) #modify to 1 channel if using grayscale images
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model_input)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('relu')(model)
    
    model = Conv2D(3, (3, 3), padding='same', kernel_initializer='he_normal')(model) #modify to 1 channel if using grayscale images
    res_img = model
    output_img = add([res_img, model_input])
    model = Model(inputs = model_input, outputs = output_img, name='VDSRCNN')
    return model

#%% a shallow custom architecture following vdsr residual learning strategy

def custom_small(input_size=(256,256,3)):     #modify to 1 channel if using grayscale images
    model_input = Input(shape=(256,256,3))  #modify to 1 channel if using grayscale images
    model = BatchNormalization()(model_input)
    model = Conv2D(8, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(16, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(32, (3, 3), padding='same', activation='relu')(model)
    model = Conv2D(64, (3, 3), padding='same', activation='relu', dilation_rate=2)(model)
    model = Conv2D(128, (3, 3), padding='same', activation='relu', dilation_rate=2)(model)
    model = Conv2D(256, (3, 3), padding='same', activation='relu', dilation_rate=2)(model)
    model = Conv2D(3, (3, 3), padding='same')(model)  #modify to 1 channel if using grayscale images
    res_img = model
    output_img = add([res_img, model_input])
    model = Model(inputs = model_input, outputs = outputimg, name='hydra')
    return model

#%% declare input parameters and data
    
image_size = 256,256 
batch = 16 
epochs = 256
input_dirname = to_dirname('data/train/')
test_dirname = to_dirname('data/test') 
x_images, y_images = load_images_train(input_dirname, image_size)
x_test, y_test = load_images_test(test_dirname, image_size)
model = vdsr() #change to custom_small if using custom architecture to train
model.summary()
gpu_model = ModelMGPU(model,2) #use this only for multi-gpu training
    
filepath='weights/' + model.name +'.best.hdf5' #modify the path for different scale versions
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
                             verbose=1, save_weights_only=True,
                             save_best_only=True, mode='min') #modify if you prefer to check validation accuracy
callbacks_list = [checkpoint]
save_model(model, 'weights/model.json') #save JSON file for model architecture
t=time.time()
optimizer = Adam(lr=0.0001) #try varying learning rates 
gpu_model.compile(optimizer=optimizer, loss=loss_mix_multi, metrics=[PSNR, ssim, ssim_multi]) # modify this model name if using non-gpu training
gpu_model.fit(x=x_images, y=y_images, 
          batch_size=batch,
          validation_data=(x_test, y_test),
          callbacks=callbacks_list,
          shuffle=True,
          epochs=epochs, verbose=1) # modify this model name if using non-gpu training
print('Training time: %s' % (time.time()-t))

#%% evaluation

x_test, y_test = load_images_test(test_dirname, image_size)
gpu_model.compile(optimizer=optimizer, loss=loss_mix_multi, 
              metrics=[PSNR, ssim, ssim_multi]) # modify this model name if using non-gpu training
eva = gpu_model.evaluate(x_test, y_test, batch_size=batch, verbose=1)
print('Evaluate: ' + str(eva))

#%% load and test the model

json_file = open('weights/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("weights/VDSRCNN.best.hdf5") #will have a different name if using the custom model
print("Loaded model from disk")
loaded_model.summary()

#%%
#PREDICT ON A SINGLE IMAGE

img = Image.open('data/test/9948_right.png')
img = img.resize((256,256)) #modify depending on resolution
img1 = image.img_to_array(img)
img_down = img.resize((256//2, 256//2)) #for scale factor of 2, modify for 3, 4, or 8
img_down = img_down.resize((256,256), Image.BICUBIC) #interpolation to upsample to original dimension
x = image.img_to_array(img_down)
x = x.astype('float32') / 255
imageio.imwrite('scaled_LR_image.png',x)
x = np.expand_dims(x, axis=0)
t=time.time()
pred = loaded_model.predict(x)
print('prediction time: %s' % (time.time()-t))
test_img = np.reshape(pred, (256,256,3)) #reshape to the original size, change channel to 1 for grayscale image
imageio.imwrite('predicted_img.png', test_img)

#%% #compute PSNR

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

d = psnr(img_down, test_img)
print(d)
print('The PSNR between the original and reconstructed image is :', d)

#%% #compute structural similarity

grayA = cv2.cvtColor(img_down, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
plt.imshow(grayA)
plt.imshow(grayB)
# Compute the Structural Similarity Index (SSIM) between the two images, ensuring that the difference image is returned

(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("The structural similarity index measure (SSIM): {}".format(score))
#this value should be close to 1 if the two images are very similar.values close to 0 says the images are very different

similarity_measure = (1-score)
print("The simialrity measure between the two images is : {}".format(similarity_measure))
# similarity measure gives zero in case of perfect match (SSIM=1); and 1 when there is no similarity (SSIM=0). 

DSSIM = 1 - (1 + score)/2 #returns 0 for similar images,and 0.5 when SSIM is 0 (dissimilar)
print("The difference in structural simialrity measured between the two images is : {}".format(DSSIM))
#This will return 0 when the two images are exactly the same, but DSSIM=1 when SSIM=-1 which corresponds with no similarity at all, and returns 1/2, when SSIM=0.

#%%
    
