import os 
from time import gmtime, strftime

import tensorflow as tf
import numpy as np
import cv2

def write_to_logs(file_name, write_string):
    """
    Read logs
    
    Args:
        file_name: the name of the log file
        write_string: the content of the log
        
    """
    time_s = strftime("%Y-%m-%d %H:%M:%S ", gmtime())
    with open (file_name, "a+") as f:
        f.write(time_s + write_string + "\n")

def read_image(image_name, feature_row, feature_col):
    """
    Read image file to tensor
    
    Args:
        image_name: the name of the image
        feature_row: the row dimension of the image, 
        feature_col: the col dimension of the image, 
        
            tensorflow need to know the size forehead in order for the batching
    Returns:
        image_tensor: the decoded image file, the value will
                      be [0,1] and the size will be [feature_row, feature_col]

    """
    image_bytes = tf.read_file(image_name)
    image_tensor = tf.image.decode_jpeg(image_bytes, channels = 3)
    image_tensor = tf.image.convert_image_dtype(image_tensor, tf.float32)
    image_tensor = tf.image.resize_images(image_tensor, feature_row, feature_col)
    return image_tensor

def display_image(image_v):
    """accept 3D or 4D numpy array. if the input is 4D, it will use the first one"""
    display_image_v = image_v
    if image_v.ndim == 4:
        display_image_v = image_v[0]
    display_image_v[:,:,[2,0]] = display_image_v[:,:,[0,2]]
    cv2.imshow("image", display_image_v)
    cv2.waitKey(100)

def save_image(image_v, loss):
    """accept 3D or 4D numpy array. if the input is 4D, it will use the first one"""
    save_image_v = image_v
    if save_image_v.ndim == 4:
        save_image_v = save_image_v[0]
    save_image_v[:,:,[2,0]] = save_image_v[:,:,[0,2]]
    save_image_v *= 255
    filename = "loss_%f.jpg" % (loss)
    np.save(filename, save_image_v)
    return
    cv2.imwrite(filename, save_image_v)
    cv2.imwrite("aaa.jpg", I)

def normalize_tensor(input_tensor):
    """normalize tensor to [-1, 1]"""
    output_tensor = ((input_tensor - tf.reduce_min(input_tensor))/ \
        (tf.reduce_max(input_tensor) - tf.reduce_min(input_tensor))) * 2 -1
    return output_tensor

def read_binary(filename, dim, dtype = tf.float32):
    """
    Read Binary file to tensor
    
    Args:
        filename: the name of the bianry file
        dim: the dimention of the binary file, assume 1D
        dtype: the type of the binary file

    Returns:
        bin_tensor: the binary tensor, it will be float32 for tensorflow

    """

    bin_file = tf.read_file(filename)
    bin_tensor = tf.decode_raw(bin_file, dtype)
    bin_tensor = tf.to_float(bin_tensor)
    bin_tensor = tf.reshape(bin_tensor,[dim])
    # bin_tensor = normalize_tensor(bin_tensor)

    return bin_tensor

def read_highd_binary(filename, shape, perm, dtype = tf.float32):
    """ 
    Read multi-dimention binary file

    Args:
        filename:
        shape:
        perm: read feature extracted from caffe
            the dimention in caffe is (channel, height, width)
            the dimention order in tensorflow is (height, width, channel)
            so perm should be [1,2,0] (the index means where the dimention come from)
        dtype:
        e.g. read_highd_binary(a.caffe_cn5, [256, 13, 13], [2,1,0])
    """

    bin_file = tf.read_file(filename)
    bin_tensor = tf.decode_raw(bin_file, dtype)
    bin_tensor = tf.to_float(bin_tensor)
    bin_tensor = tf.reshape(bin_tensor, shape)
    bin_tensor = tf.transpose(bin_tensor, perm= perm)
    return bin_tensor


def read_multiple_binary(filenames, binary_num, dim, dtype = tf.float32):
    """
    Read Binary file to tensor
    
    Args:
        filename: list of the name of the bianry file
        binary_num: the name of the binary file
        dim: the dimention of the binary file, assume 1D
        dtype: the type of the binary file

    Returns:
        bin_tensor: the binary tensor, it will be list of float32 for tensorflow

    """
    multi_binary = list()
    multi_filename = list()
    for i in range(binary_num):
        bin_tensor = read_binary(filenames[i], dim)
        bin_tensor = tf.expand_dims(bin_tensor,0)
        multi_binary.append(bin_tensor)
        new_filename = tf.expand_dims(filenames[i], 0)
        multi_filename.append(new_filename)

    concat_binary = tf.concat(0,multi_binary)
    concat_filenames = tf.concat(0, multi_filename)
    return concat_binary, concat_filenames

def read_multiple_highd_binary(filenames, binary_num, shape, perm, dtype = tf.float32):
    """
    Read Multiply High Dimentional Binary file to tensor
    
    def read_highd_binary(filename, shape, perm, dtype = tf.float32):
    Args:
        filename: list of the name of the bianry file
        binary_num: the name of the binary file
        shape: a list, shape of each binary file
        perm: a list, order of channel each binary file
        dtype: the type of the binary file

    Returns:
        bin_tensor: the binary tensor, it will be list of float32 for tensorflow

    """
    multi_binary = list()
    multi_filename = list()
    for i in range(binary_num):
        bin_tensor = read_highd_binary(filenames[i], shape, perm, dtype)
        bin_tensor = tf.expand_dims(bin_tensor,0)
        multi_binary.append(bin_tensor)
        new_filename = tf.expand_dims(filenames[i], 0)
        multi_filename.append(new_filename)

    concat_binary = tf.concat(0,multi_binary)
    concat_filenames = tf.concat(0, multi_filename)
    return concat_binary, concat_filenames

def tower_loss(tower_name, scope):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
    
  with tf.control_dependencies([loss_averages_op]):
    total_loss = tf.identity(total_loss)

  return total_loss


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)

  return average_grads

def define_graph_config(fraction):
    """Define the GPU usage"""
    config_proto =  tf.ConfigProto()
    config_proto.gpu_options.per_process_gpu_memory_fraction = fraction
    config_proto.allow_soft_placement=True
    config_proto.log_device_placement=False
    config_proto.gpu_options.allow_growth=True
    return config_proto

def dense_to_one_hot_numpy(labels_dense, num_classes):
    """
    Numpy version:
    Convert class labels from scalars to one-hot vectors.
    """
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    index = (index_offset + labels_dense.ravel()).astype(int)
    labels_one_hot.flat[index] = 1
    return labels_one_hot

def one_hot_to_dense_numpy(labels_one_hot):
    """
    Numpy version:
    Convert one-hot vectors to dense label
    """
    labels_dense = np.argmax(labels_one_hot, axis = 1)
    return labels_dense

def dense_to_one_hot(labels_batch, num_classes):
    """
    Tensorflow version:
    Convert class labels from scalars to one-hot vectors.
    """
    sparse_labels = tf.reshape(labels_batch, [-1, 1])
    derived_size = tf.shape(labels_batch)[0]
    indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    outshape = tf.pack([derived_size, num_classes])
    labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
    return labels

def cal_accuracy(label_list, classify_res):
    """ calculate the accuracy"""
    assert(len(label_list) == len(classify_res))
    right_count = 0
    for i in range(len(label_list)):
        if (label_list[i] == classify_res[i]):
            right_count += 1
    return right_count / float(len(label_list))

def cal_percision(label_list, classify_res):
    """ calculate the percision"""
    assert(len(label_list) == len(classify_res))
    true_positive_count = 0.0
    false_positive_count = 0.0
    for i in range(len(label_list)):
        if classify_res[i] == True:
            if label_list[i] == True:
                true_positive_count += 1
            else:
                false_positive_count += 1
    if (true_positive_count + false_positive_count == 0):
        percision = 0.0
    else:
        percision = true_positive_count/(true_positive_count + false_positive_count)
    return percision

