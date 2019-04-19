import os

import cv2
import numpy as np


data_dir = "../../data/mnist_data/MNIST/"


class Mnist(object):
    def __init__(self, is_train, data_num=None):
        train_image_name = "train-images-idx3-ubyte"
        train_label_name = "train-labels-idx1-ubyte"
        test_image_name = "t10k-images-idx3-ubyte"
        test_label_name = "t10k-labels-idx1-ubyte"

        if is_train:
            self.image = self.read_image(data_dir + train_image_name)
            self.label = self.read_label(data_dir + train_label_name)
        else:
            self.image = self.read_image(data_dir + test_image_name)
            self.label = self.read_label(data_dir + test_label_name)

        if data_num is not None:
            data_num = min(data_num, self.label.shape[0])
            self.image = self.image[:data_num]
            self.label = self.label[:data_num]

        self.file_size = self.label.shape[0]
        self.curr_index = 0

    def load_data(self, batch_size):
        start_index = self.curr_index
        end_index = start_index + batch_size
        if end_index <= self.file_size:
            image = self.image[start_index: end_index]
            label = self.label[start_index: end_index]
        else:
            image = self.image[start_index:]
            label = self.label[start_index:]
            end_index = end_index - self.file_size
            image = np.vstack((image, self.image[:end_index]))
            label = np.append(label, self.label[:end_index])

            self.shuffle_data()

        self.curr_index = end_index
        return image, label

    def shuffle_data(self):
        index = np.arange(self.file_size)
        np.random.shuffle(index)
        np.take(self.image, index, axis=0, out=self.image)
        np.take(self.label, index, axis=0, out=self.label)

    def read_image(self, file_name):
        with open(file_name, "rb") as f:
            magic, size, rows, cols = np.fromfile(f, ">4I", 1)[0]
            image_data = np.fromfile(f, np.uint8)
            images = np.empty((size, 32, 32, 3), np.float32)
            for i in range(size):
                image = image_data[i * rows *
                                   cols: (i + 1) * rows * cols] / 255.0
                image = np.reshape(image, (rows, cols))
                images[i, :, :, :] = np.tile(np.expand_dims(cv2.resize(
                    image.astype(np.float32), (32, 32)), 2), (1, 1, 1, 3))
            return images

    def read_label(self, file_name):
        with open(file_name, "rb") as f:
            magic, size = np.fromfile(f, ">2I", 1)[0]
            labels = np.fromfile(f, np.uint8)
            return labels


if __name__ == "__main__":
    mnist = Mnist(is_train=True, data_num=600)
    for i in range(10000):
        image, label = mnist.load_data(10)
        print(label)
        assert(label.shape[0] == 10)
        assert(image.shape[3] == 3)
        if i == 10:
            exit(1)
