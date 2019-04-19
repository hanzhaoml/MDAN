from TensorflowToolbox.utility import file_io
import cv2
import numpy as np


data_dir = "../../data/mnist_data/MNISTM/"


class MnistM(object):
    def __init__(self, is_train, data_num=None):
        train_file_name = data_dir + "mnist_m_train_labels.txt"
        test_file_name = data_dir + "mnist_m_test_labels.txt"
        train_data_dir = data_dir + "mnist_m_train/"
        test_data_dir = data_dir + "mnist_m_test/"

        if is_train:
            self.image, self.label = self.read_data(
                train_data_dir, train_file_name, data_num)
        else:
            self.image, self.label = self.read_data(
                test_data_dir, test_file_name, data_num)

        if data_num is not None:
            data_num = min(data_num, self.label.shape[0])
            self.image = self.image[:data_num]
            self.label = self.label[:data_num]

        self.file_size = self.label.shape[0]
        self.curr_index = 0

    def read_data(self, data_dir, file_name, data_num):
        file_list = file_io.read_file(file_name)
        data_len = min(data_num, len(file_list))
        images = np.empty((data_len, 32, 32, 3), np.float32)
        labels = np.empty((data_len), np.uint8)
        for i in range(data_len):
            image_name, label = file_list[i].split(" ")
            images[i, :, :, :] = cv2.imread(data_dir + image_name) / 255.0
            labels[i] = int(label)

        return images, labels

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


if __name__ == "__main__":
    mnist = MnistM(True, 60)
    for i in range(10):
        image, label = mnist.load_data(10)
        for i in range(10):
            print(label[i])
        # exit(1)
        assert(label.shape[0] == 10)
        assert(image.shape[0] == 10)
