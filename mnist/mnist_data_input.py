import cv2
import tensorflow as tf
import numpy as np

import mnist
import mnist_m
import svhn
import synthdigits


class DataInput(object):
    def __init__(self, model_params, mnist_type, phase, is_train):
        self.batch_size = model_params["batch_size"]
        max_data_num = model_params["max_data_num"]

        self.is_train = is_train
        if mnist_type == "MNIST":
            self.data_type = mnist.Mnist(is_train, max_data_num)
        elif mnist_type == "MNIST_M":
            self.data_type = mnist_m.MnistM(is_train, max_data_num)
        elif mnist_type == "SVHN":
            self.data_type = svhn.Svhn(is_train, max_data_num)
        elif mnist_type == "SYNTHDIGITS":
            self.data_type = synthdigits.SynthDigits(is_train, max_data_num)

        self.file_size = self.data_type.file_size

    def get_arg_dict(self, model_params):
        arg_dict = dict()

        arg_dict["feature"] = dict()
        arg_dict["label"] = dict()
        arg_dict["mask"] = dict()

        for key in model_params:
            if "data_arg" in key:
                _, domain, field = key.split(".")
                arg_dict[domain][field] = model_params[key]
        arg_dict_list = list()
        arg_dict_list.append(arg_dict["feature"])
        arg_dict_list.append(arg_dict["label"])
        arg_dict_list.append(arg_dict["mask"])
        return arg_dict_list

    def load_data(self):
        image, label = self.data_type.load_data(self.batch_size)

        return image, label

    def center_crop(self, img):
        i_height, i_width, i_cha = img.get_shape().as_list()
        ccrop_size = [30, 30]
        offset_height = int((i_height - ccrop_size[0]) / 2)
        offset_width = int((i_width - ccrop_size[1]) / 2)
        img = tf.image.crop_to_bounding_box(img,
                                            offset_height, offset_width, ccrop_size[0], ccrop_size[1])
        return img

    def add_data_arg(self, imgs):
        if self.is_train:
            imgs = tf.map_fn(
                lambda img: tf.image.random_brightness(img, 0.2),
                imgs)

            imgs = tf.map_fn(lambda img:
                             tf.image.random_contrast(img, 0.5, 1.5),
                             imgs)

            imgs = tf.map_fn(lambda img:
                             tf.random_crop(img, [30, 30, 3]),
                             imgs)
        else:
            imgs = tf.map_fn(lambda img:
                             self.center_crop(img),
                             imgs)
        return imgs

    def get_input(self):
        pass

    def get_label(self):
        pass


if __name__ == "__main__":
    """ example of running the code"""
    model_params = {"batch_size": 10}
    data_input = DataInput(model_params, "SYNTHDIGITS", None, True)

    for i in range(10):
        image, label = data_input.load_data()
        for i in range(10):
            print(label[i])
            cv2.imshow("img", image[i])
            cv2.waitKey(0)
