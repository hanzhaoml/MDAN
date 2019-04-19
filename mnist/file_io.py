import os
import random
import numpy as np
import importlib


def get_listfile(image_dir, extension=".jpg"):
    if not image_dir.endswith("/"):
        image_dir = image_dir + "/"

    image_list = os.listdir(image_dir)
    image_list = [image_dir + image for image in image_list if image.endswith(extension)]
    return image_list


def get_dir_list(frame_dir):
    if not frame_dir.endswith("/"):
        frame_dir = frame_dir + "/"

    dir_list = os.listdir(frame_dir)
    dir_list = [frame_dir +
                image_dir for image_dir in dir_list if os.path.isdir(frame_dir + image_dir)]
    return dir_list


def delete_last_empty_line(s):
    end_index = len(s) - 1
    while(end_index >= 0 and (s[end_index] == "\n" or s[end_index] == "\r")):
        end_index -= 1
    s = s[:end_index + 1]
    return s


def read_file(file_name):
    with open(file_name, "r") as f:
        s = f.read()
        s = delete_last_empty_line(s)
        s_l = s.split("\n")
        for i, l in enumerate(s_l):
            if l.endswith("\r"):
                s_l[i] = s_l[i][:-1]
    return s_l


def save_file(string_list, file_name, shuffle_data=False):
    if (shuffle_data):
        random.shuffle(string_list)

    with open(file_name, "w") as f:
        if not len(string_list):
            f.write("")
        else:
            file_string = '\n'.join(string_list)
            if (file_string[-1] != "\n"):
                file_string += "\n"
            f.write(file_string)


def get_file_length(file_name):
    with open(file_name, 'r') as f:
        s = f.read()
        s_l = s.split("\n")
        total_len = len(s_l)
    return total_len


def save_numpy_array(numpy_array, file_name):
    numpy_array.tofile(file_name)


def remove_extension(file_name):
    index = file_name.rfind(".")
    if (index == -1):
        return file_name
    else:
        return file_name[0:index]


def import_module_class(module_name, class_name=None):
    module = importlib.import_module(module_name)
    if class_name == None:
        return module
    else:
        return getattr(module, class_name)


def check_exist(file_name):
    """
    Args:
        file_name: file name of the file list
                    i.e.: train_list.txt
    """
    file_list = read_file(file_name)
    for i, f in enumerate(file_list):
        f_l = f.split(" ")
        for ff in f_l:
            is_exist = os.path.exists(ff)
            if not is_exist:
                raise OSError("In %s, row: %d, "
                              "%s does not exist" % (file_name, i, ff))


def save_string(input_string, file_name):
    if os.path.exists(file_name):
        mode = "a"
    else:
        mode = "w"
    if not input_string.endswith("\n"):
        input_string += "\n"

    with open(file_name, mode) as f:
        f.write(input_string)
