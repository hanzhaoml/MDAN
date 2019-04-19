import scipy.io as sio
import cv2
import numpy as np

data_dir = "../../data/mnist_data/SynthDigits/"


class SynthDigits(object):
    def __init__(self, is_train, data_num=None):
        #data_dir = "/Users/Geoff/Documents/my_git/data/Four_Dataset/SynthDigits/"
        train_file_name = "synth_train_32x32.mat"
        test_file_name = "synth_test_32x32.mat"
        if is_train:
            self.image, self.label = self.read_image(data_dir + train_file_name)
        else:
            self.image, self.label = self.read_image(data_dir + test_file_name)

        if data_num is not None:
            data_num = min(data_num, self.label.shape[0])
            self.image = self.image[:data_num]
            self.label = self.label[:data_num]
        self.image = self.image.astype(np.float32) / 255.0
        self.label = self.label.astype(np.uint8)
        self.file_size = self.label.shape[0]
        self.curr_index = 0

    def read_image(self, file_name):
        mat_contents = sio.loadmat(file_name)
        images = np.transpose(mat_contents["X"], (3, 0, 1, 2))
        labels = np.squeeze(mat_contents["y"])

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
    mnist = SynthDigits(True, 6000)
    for i in range(10):
        image, label = mnist.load_data(10)
        print(label)
        assert(label.shape[0] == 10)
        assert(image.shape[0] == 10)
