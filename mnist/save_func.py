import time
import os

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

def add_train_var():
    """ add all trainable variable to summary"""
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)


def add_loss(loss_scope = 'losses'):
    """ add all losses to summary """
    for l in tf.get_collection(loss_scope):
        tf.summary.scalar(l.op.name, l)

def add_image(image_collection, image_num = -1):
    """
    Args:
        image_num: the number of images to save
                   if it is set to be -1, the whole batch will be saved
    """
    for var in tf.get_collection(image_collection):
        if image_num == -1:
            image_num = var.get_shape()[0]
        tf.summary.image(var.op.name, var, image_num)

def restore_model(sess, saver, model_dir, model_name=None):
    """ restore model:
            if model_name is None, restore the last one
    """
    if model_name is None:
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.all_model_checkpoint_paths[-1]:
            print("restore " + ckpt.all_model_checkpoint_paths[-1])
            saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
        else:
            print('no check point')
    else:
        model_path = os.path.join(model_dir, model_name)
        print("restore " + model_path)
        saver.restore(sess, model_path)
	
def save_model(sess, saver, model_dir, iteration):
    """ save the current model"""
    if not model_dir.endswith("/"):
        model_dir += "/"

    curr_time = time.strftime("%Y%m%d_%H%M")
    model_name = model_dir + curr_time + \
                            '_iter_' + str(iteration) + '_model.ckpt'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    saver.save(sess, model_name)

def add_value_sum(summary_writer, value, name, iteration):
    """ add python value to tensorboard """
    summary = tf.Summary(value = [tf.Summary.Value(tag = name, simple_value = value)])	
    summary_writer.add_summary(summary, iteration)


def group_mv_ops(train_op, moving_average_decay, global_step):
    """ group all the operations 
    Args:
    """	
    # batchnorm_updates = tf.get_collection(FLAGS.bn_collection)
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    # batchnorm_vars = tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)	
    variables_to_average = tf.trainable_variables()
                                                    # batchnorm_vars)
            
    variables_averages_op = variable_averages.apply(variables_to_average)
    # batchnorm_updates_op = tf.group(*batchnorm_updates)
    all_op = tf.group(train_op, variables_averages_op)

    return all_op


def partial_restore(pretrain_model_name, sess, exclude_str_list=[]):
    var_dict = dict()
    for var in tf.trainable_variables():
        var_dict[var.op.name] = var
    reader = pywrap_tensorflow.NewCheckpointReader(pretrain_model_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    restore_dict = {}
    for key in sorted(var_to_shape_map):
        if key in var_dict:
            exclude = [ex_str for ex_str in exclude_str_list if ex_str in key]
            if exclude:
               continue
            restore_dict.update({key: var_dict[key]})
    
    if restore_dict:
        saver = tf.train.Saver(restore_dict)
        restore_model(sess, saver, "", pretrain_model_name)
        print('Successfully restored {} vars in {}.'
              .format(len(restore_dict), pretrain_model_name))
    else:
        print('no variable to restore in {}'.format(pretrain_mode_name))
