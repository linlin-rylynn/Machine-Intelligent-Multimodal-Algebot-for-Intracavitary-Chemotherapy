import colorsys
import copy
import time

import cv2
import numpy as np
from PIL import Image

from Unet.unet import Unet as unet
from Unet.Unet_utils.utils1 import cvtColor, preprocess_input, resize_image, show_config


class Unet(object):
    _defaults = {
        "model_path"        : 'log/yourfiles.h5',
        "num_classes"       : 5,
        "backbone"          : "vgg",
        "input_shape"       : [512, 512],
        "mix_type"          : 0,
    }

    #---------------------------------------------------#
    #   Initialize UNET
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

    #---------------------------------------------------#
    #  Segmentation area color mask of US images
    #---------------------------------------------------#             
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (255, 140, 0), (255, 105, 180), (0, 0, 0),(100, 79, 131)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)
    #---------------------------------------------------#
    #   Load the model
    #---------------------------------------------------#
    def generate(self):
        #-------------------------------#
        #   Load model and weights
        #-------------------------------#
        self.model = unet([self.input_shape[0], self.input_shape[1], 3], self.num_classes, self.backbone)

        self.model.load_weights(self.model_path)
        print('{} model loaded.'.format(self.model_path))

    #---------------------------------------------------#
    #   Detect image
    #---------------------------------------------------#
    def detect_image(self, image, count=False, name_classes=None):
        #---------------------------------------------------------#
        #   Convert the image to RGB to prevent errors with grayscale images
        #   The code only supports RGB image prediction, all other types will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------#
        #   Make a copy of the input image for later drawing
        #---------------------------------------------------#
        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   Add gray bars to the image for non-distorted resize
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        #---------------------------------------------------------#
        #   Normalization + add batch_size dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        #---------------------------------------------------#
        #   Feed the image into the network for prediction
        #---------------------------------------------------#
        pr = self.model.predict(image_data)[0]
        #---------------------------------------------------#
        #   Crop out the gray bars
        #---------------------------------------------------#
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        #---------------------------------------------------#
        #   Resize the image
        #---------------------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
        #---------------------------------------------------#
        #   Get the class of each pixel
        #---------------------------------------------------#
        pr = pr.argmax(axis=-1)
        
        #---------------------------------------------------------#
        #   Counting
        #---------------------------------------------------------#
        if count:
            classes_nums        = np.zeros([self.num_classes])
            total_points_num    = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num     = np.sum(pr == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if self.mix_type == 0:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   Convert the new image to Image format
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))
            #------------------------------------------------#
            #   Blend the new image with the original image
            #------------------------------------------------#
            image   = Image.blend(old_img, image, 0.5)

        elif self.mix_type == 1:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   Convert the new image to Image format
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            #------------------------------------------------#
            #   Convert the new image to Image format
            #------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))

        return image

    def get_FPS(self, image, test_interval):
        #---------------------------------------------------------#
        #   Convert the image to RGB to prevent errors with grayscale images
        #   The code only supports RGB image prediction, all other types will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Add gray bars to the image for non-distorted resize
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        #---------------------------------------------------------#
        #   Normalization + add batch_size dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        #---------------------------------------------------#
        #   Feed the image into the network for prediction
        #---------------------------------------------------#
        pr = self.model.predict(image_data)[0]
        #--------------------------------------#
        #   Crop out the gray bars
        #--------------------------------------#
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        #---------------------------------------------------#
        #   Get the class of each pixel
        #---------------------------------------------------#
        pr = pr.argmax(axis=-1).reshape([self.input_shape[0],self.input_shape[1]])
                
        t1 = time.time()
        for _ in range(test_interval):
            #---------------------------------------------------#
            #   Feed the image into the network for prediction
            #---------------------------------------------------#
            pr = self.model.predict(image_data)[0]
            #--------------------------------------#
            #   Crop out the gray bars
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   Get the class of each pixel
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1).reshape([self.input_shape[0],self.input_shape[1]])

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
        
    def get_miou_png(self, image):
        #---------------------------------------------------------#
        #   Convert the image to RGB to prevent errors with grayscale images
        #   The code only supports RGB image prediction, all other types will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   Add gray bars to the image for non-distorted resize
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        #---------------------------------------------------------#
        #   Normalization + add batch_size dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)

        #--------------------------------------#
        #   Feed the image into the network for prediction
        #--------------------------------------#
        pr = self.model.predict(image_data)[0]
        #--------------------------------------#
        #   Crop out the gray bars
        #--------------------------------------#
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        #--------------------------------------#
        #   Resize the image
        #--------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
        #---------------------------------------------------#
        #   Get the class of each pixel
        #---------------------------------------------------#
        pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image