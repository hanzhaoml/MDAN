import tensorflow as tf
from tensorflow.python.training import moving_averages


FLAGS = tf.app.flags.FLAGS


def _variable_on_cpu(name, shape, initializer, trainable=True):
    """Helper to create a Variable stored on CPU memory.

    Args:
            name: name of the variable
            shape: list of ints
            initializer: initializer for Variable

    Returns:
            Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return var


def _variable_with_weight_decay(name, shape, wd=0.0,
                                initializer=tf.contrib.layers.xavier_initializer()):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a xavier initialization.
    A weight decay is added only if one is specified.

    #Args:
            name: name of the variable
            shape: list of ints
            wd: add L2Loss weight decay multiplied by this float. If None, weight
                    decay is not added for this Variable.

    Returns:
            Variable Tensor
    """
    var = _variable_on_cpu(name, shape, initializer)
    # print("change var")
    # var = tf.Variable(tf.truncated_normal(shape, mean= 0.0, stddev = 1.0), name = name)
    if wd != 0.0:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_decay_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _conv2d(x, w, b, strides, padding, data_format=None):
    return tf.nn.bias_add(tf.nn.conv2d(x, w, strides=strides, padding=padding,
                                       data_format=data_format), b, data_format=data_format)


def _conv3d(x, w, b, strides=[1, 1, 1, 1, 1], padding='SAME'):
    return tf.nn.bias_add(tf.nn.conv3d(x, w, strides=strides, padding=padding), b)


def add_leaky_relu(hl_tensor, leaky_param, layer_name=None):
    with tf.name_scope(layer_name, 'relu'):
        leaky_relu = tf.maximum(hl_tensor, tf.multiply(leaky_param, hl_tensor))
    return leaky_relu


def _deconv2d(x, w, b, output_shape, strides, padding, data_format):
    return tf.nn.bias_add(tf.nn.conv2d_transpose(
        x, w, output_shape, strides, padding, data_format), b)


def _add_leaky_relu(hl_tensor, leaky_param):
    """ add leaky relu layer
        Args:
            leaky_params should be from 0.01 to 0.1
    """
    return tf.maximum(hl_tensor, tf.multiply(leaky_param, hl_tensor))


def _max_pool(x, ksize, strides, data_format):
    """ 2d pool layer"""
    pool = tf.nn.max_pool(x, ksize=ksize, strides=strides,
                          padding='VALID', data_format=data_format)
    return pool


def _max_pool3(x, ksize, strides, name):
    """ 3d pool layer"""
    pool = tf.nn.max_pool3d(x, ksize=ksize, strides=strides,
                            padding='VALID', name=name)
    return pool


def _avg_pool3(x, ksize, strides, name):
    """ 3d average pool layer """
    pool = tf.nn.avg_pool3d(x, ksize=ksize, strides=strides,
                            padding='VALID', name=name)
    return pool


def triplet_loss(infer, labels, radius=2.0):
    """
    Args:
        infer: inference concatenate together with 2 * batch_size
        labels: 0 or 1 with batch_size
        radius:
    Return:
        loss: triplet loss
    """

    feature_1, feature_2 = tf.split(0, 2, infer)

    feature_diff = tf.reduce_sum(tf.square(feature_1 - feature_2), 1)
    feature_list = tf.dynamic_partition(feature_diff, labels, 2)

    pos_list = feature_list[1]
    neg_list = (tf.maximum(0.0, radius * radius - feature_list[0]))
    full_list = tf.concat(0, [pos_list, neg_list])
    loss = tf.reduce_mean(full_list)

    return loss


def l1_reg(input_tensor, weights):
    l1_reg_loss = tf.multiply(tf.reduce_sum(tf.abs(input_tensor)), weights, name="l1_reg_loss")
    tf.add_to_collection('losses', l1_reg_loss)


def l2_loss(infer, label, loss_type, layer_name):
    """
    Args:
        loss_type: 'SUM', 'MEAN'
            'SUM' uses reduce_sum
            'MEAN' uses reduce_mean
    """
    assert(loss_type == 'SUM' or loss_type == 'MEAN')
    with tf.variable_scope(layer_name):
        if loss_type == 'SUM':
            loss = tf.reduce_sum(tf.square(infer - label))
        else:
            loss = tf.reduce_mean(tf.square(infer - label))

    return loss


def l1_loss(infer, label, loss_type, layer_name):
    """
    Args:
        loss_type: 'SUM', 'MEAN'
            'SUM' uses reduce_sum
            'MEAN' uses reduce_mean
    """
    assert(loss_type == 'SUM' or loss_type == 'MEAN')
    with tf.variable_scope(layer_name):
        if loss_type == 'SUM':
            loss = tf.reduce_sum(tf.abs(infer - label))
        else:
            loss = tf.reduce_mean(tf.abs(infer - label))

    return loss


def x_entropy_loss(infer, label, layer_name):
    """
    Args:
        infer:
        label:
        layer_name:
    """
    with tf.variable_scope(layer_name):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=infer, labels=label))
    return loss


def image_l2_loss(infer, label, layer_name):
    """
    Args:
        infer: [batch_size, height, width, channel]
        label: [batch_size, height, width, channel]
    Return:
        for each batch: sum([infer - label] ^ 2), then
            calculate the mean for the entire batch
    """
    with tf.variable_scope(layer_name):
        l2_loss = tf.reduce_mean(tf.reduce_sum(tf.square(infer - label),
                                               [1, 2, 3]), name='l2_loss')
    return l2_loss


def image_l1_loss(infer, label, layer_name):
    with tf.variable_scope(layer_name):
        l1_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(infer - label),
                                               [1, 2, 3]), name="l1_loss")
    return l1_loss


def count_diff(infer, label, layer_name):
    with tf.variable_scope(layer_name):
        img_diff = tf.reduce_mean(tf.abs(tf.reduce_sum((infer - label),
                                                       [1, 2, 3])), name="count_diff")
    return img_diff


def huber_loss(infer, label, epsilon, reduction, weights, layer_name):
    """
    Args:
        infer
        label
        epsilon
        reduction: "MEAN" or "SUM"
        weights: 1 or a vec
        layer_name
    """
    if reduction == "MEAN":
        reduction_method = tf.losses.Reduction.MEAN
    elif reduction == "SUM":
        reduction_method = tf.losses.Reduction.SUM
    else:
        raise NotImplementedError("reduction {} is not implemented".format(reduction))

    with tf.variable_scope(layer_name):
        hloss = tf.losses.huber_loss(labels=label, predictions=infer, delta=epsilon,
                                     weights=weights, reduction=reduction_method)
    return hloss


def convolution_2d_layer(inputs, filters, kernel_size, kernel_stride, padding,
                         data_format='NCHW', bn=False, is_train=False, leaky_params=None,
                         wd=0.0, layer_name='conv2d'):
    """
    Args:
        inputs:
        filters: integer
        kernel_size: [height, width]
        kernel_stride: [height, width]
        padding: "SAME" or "VALID"
        data_format: 'NCHW'.
        bn: True/False, if do batch norm.
        leaky_params: None will be no relu.
        wd: weight decay params.
        layer_name: 
    """
    with tf.variable_scope(layer_name):
        input_shape = inputs.get_shape().as_list()

        kerner_initializer = tf.contrib.layers.xavier_initializer()
        bias_initializer = tf.zeros_initializer
        if data_format == "NCHW":
            input_channel = input_shape[1]
            stride = [1, 1, kernel_stride[0], kernel_stride[1]]
        elif data_format == "NHWC":
            input_channel = input_shape[3]
            stride = [1, kernel_stride[0], kernel_stride[1], 1]
        else:
            raise NotImplementedError

        weights = _variable_with_weight_decay('weights',
                                              kernel_size + [input_channel, filters],
                                              wd, kerner_initializer)

        biases = _variable_on_cpu('biases', filters, bias_initializer)
        conv = _conv2d(inputs, weights, biases, stride, padding, data_format)

        if bn:
            axis = -1
            if data_format == "NCWH":
                axis = 1

            conv = batch_norm_layer(conv, axis, is_train, True)

        if leaky_params is not None:
            conv = add_leaky_relu(conv, leaky_params)

    return conv


def fully_connected_layer(x, filters, leaky_params=None, wd=0.0, layer_name="fc"):
    """
    Args:
        x
        output_num 
        wd
        layer_num
    """
    #input_shape = x.get_shape().as_list()
    # if len(input_shape) > 2:
    #    x = tf.reshape(x, [input_shape[0], -1])

    #input_shape = x.get_shape().as_list()

    with tf.variable_scope(layer_name):
        input_shape = x.get_shape().as_list()
        input_channel = input_shape[1]
        if len(input_shape) > 2:
            mul = 1
            for m in input_shape[1:]:
                mul *= m
            x = tf.reshape(x, [-1, mul])
            input_channel = mul

        kerner_initializer = tf.contrib.layers.xavier_initializer()
        bias_initializer = tf.zeros_initializer

        weights = _variable_with_weight_decay('weights',
                                              [input_channel, filters],
                                              wd, kerner_initializer)

        biases = _variable_on_cpu('biases', filters, bias_initializer)
        fc = tf.matmul(x, weights) + biases

        if leaky_params is not None:
            fc = add_leaky_relu(fc, leaky_params)

    return fc


def deconvolution_2d_layer(inputs, filters, kernel_size, strides, padding, output_size,
                           data_format, bn, is_train, leaky_params, wd, layer_name):
    """
    Args:
        inputs:
        filters: number of output channel.
        kernel_size: [height, width]
        stride: [height, width]
        padding: "SAME" or "VALID"
        output_size: [h, w]
        data_format: "NCHW"
        bn: True/False, if do batch norm.
        leaky_params: None will be no relu.
        wd: weight decay params.
        layer_name: 
    """
    with tf.variable_scope(layer_name):
        input_shape = inputs.get_shape().as_list()
        batch_size = tf.shape(inputs)[0]

        kerner_initializer = tf.contrib.layers.xavier_initializer()
        bias_initializer = tf.zeros_initializer

        if data_format == "NCHW":
            input_channel = input_shape[1]
            strides = [1, 1, strides[0], strides[1]]
            output_shape = [batch_size, filters, output_size[0], output_size[1]]
        elif data_format == "NHWC":
            input_channel = input_shape[3]
            strides = [1, strides[0], strides[1], 1]
            output_shape = [batch_size, output_size[0], output_size[1], filters]
        else:
            raise NotImplementedError

        kernel_initializer = tf.contrib.layers.xavier_initializer()
        bias_initializer = tf.zeros_initializer

        weights = _variable_with_weight_decay('weights',
                                              kernel_size + [filters, input_channel],
                                              wd, kerner_initializer)

        biases = _variable_on_cpu('biases', filters, bias_initializer)

        deconv = _deconv2d(inputs, weights, biases, output_shape, strides, padding, data_format)

        if bn:
            axis = -1
            if data_format == "NCWH":
                axis = 1

            deconv = batch_norm_layer(deconv, axis, is_train, True)

        if leaky_params is not None:
            deconv = add_leaky_relu(deconv, leaky_params)

    return deconv


def deconvolution_2d_layer2(inputs, filters, kernel_size, strides, padding, data_format,
                            bn, is_train, leaky_params, wd, layer_name):
    """
    Args:
        inputs:
        filters: number of output channel.
        kernel_size: [height, width]
        kernel_stride: [height, width]
        padding: "SAME" or "VALID"
        data_format: "NCHW"
        bn: True/False, if do batch norm.
        leaky_params: None will be no relu.
        wd: weight decay params.
        layer_name: 
        # output_shape: [batch_size, height, width, channel]
        # padding: "SAME" or "VALID"
        # wd: weight decay params
        # layer_name: 
    """
    with tf.variable_scope(layer_name):
        kernel_initializer = tf.contrib.layers.xavier_initializer()
        bias_initializer = tf.zeros_initializer

        if data_format == "NCHW":
            data_format = 'channels_first'
        else:
            data_format = 'channels_last'

        deconv = tf.layers.conv2d_transpose(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=None,
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            trainable=True,
            name='deconv',
            reuse=None
        )
        scope = tf.get_variable_scope().name

        if wd != 0.0:
            kernel = tf.get_default_graph().get_tensor_by_name(scope + '/deconv/kernel:0')
            weight_decay = tf.multiply(tf.nn.l2_loss(kernel), wd, name='weight_decay_loss')
            tf.add_to_collection('losses', weight_decay)

        if bn:
            axis = -1
            if data_format == "NCWH":
                axis = 1

            deconv = batch_norm_layer(deconv, axis, is_train, True)

        if leaky_params is not None:
            deconv = add_leaky_relu(deconv, leaky_params)

    return deconv


def maxpool_2d_layer(x, kernel_shape, kernel_stride, data_format, layer_name):
    """
    Args:
        x
        kernel_shape: [height, weights]
        kernel_stride: [height, weights]
    """

    with tf.variable_scope(layer_name):
        if data_format == "NCHW":
            kernel_shape = [1, 1, kernel_shape[0], kernel_shape[1]]
            stride = [1, 1, kernel_stride[0], kernel_stride[1]]
        elif data_format == "NHWC":
            kernel_shape = [1, kernel_shape[0], kernel_shape[1], 1]
            stride = [1, kernel_stride[0], kernel_stride[1], 1]
        else:
            raise NotImplementedError

        max_pool = _max_pool(x, kernel_shape, stride, data_format)
    return max_pool


def res_layer(x, kernel_shape, kernel_stride, padding, wd, layer_name, repeat_num, leaky_param=0.01, is_train=None):
    """
    Args:
        x
        kernel_shape: [height, weights, input_channel, ouput_channel]
        kernel_stride: [height, weights]
        padding: SAME or VALID
        is_train: a tensor indicate is train or not
    """
    with tf.variable_scope(layer_name):
        conv = tf.identity(x)
        for i in xrange(repeat_num):
            conv = convolution_2d_layer(conv, kernel_shape,
                                        kernel_stride, padding, wd, "_%d" % i)
            if is_train is not None:
                conv = _batch_norm(conv, is_training=is_train)
            conv = add_leaky_relu(conv, leaky_param)

        final_conv = tf.add(conv, x, 'res_connect')
    return final_conv


def batch_norm_layer(x, axis, is_train, renorm, name='bn'):
    if axis == -1:
        data_format = 'NHWC'
    else:
        data_format = 'NCHW'
    bn = tf.contrib.layers.batch_norm(
            x, is_training=is_train, fused=True, data_format=data_format, renorm=False)

    return bn


def res_pad(x, input_channel, output_channel, layer_name):
    """
    Args:
        x
        input_channel: a number
        output_channel: a number
        layer_name
    """
    with tf.variable_scope(layer_name):
        forward_pad = (output_channel - input_channel) // 2
        backward_pad = output_channel - input_channel - forward_pad
        x_pad = tf.pad(x, [[0, 0], [0, 0], [0, 0], [forward_pad, backward_pad]])
    return x_pad


def copy_layer(x, layer_handle, repeat_num, layer_name, *params):
    """
    Args:
        x
        layer_handle: function handler
        repeat_num: the number of repeat
        layer_name
        params: parameters for the function
    """
    for i in xrange(repeat_num):
        with tf.variable_scope(layer_name + "_%d" % i):
            x = layer_handle(x, *params)
    return x


def unpooling_layer(x, output_size, layer_name):
    """ Bilinear Interpotation resize 
    Args:
        x
        output_size [image_height, image_width]
        layer_name 
    """
    with tf.variable_scope(layer_name):
        return tf.image.resize_images(x, output_size[0], output_size[1])


def atrous_convolution_layer(inputs, filters, kernel_size, rate, padding="SAME", 
                             data_format="NCHW", 
                             bn=False, is_train=False, leaky_params=None, wd=0.0,
                             layer_name='atrous_conv2d'):
    """
    Args:
        x
        kernel_shape
        rate
        padding
        wd
        layer_name
    """
    with tf.variable_scope(layer_name):
        input_shape = inputs.get_shape().as_list()

        kerner_initializer = tf.contrib.layers.xavier_initializer()
        bias_initializer = tf.zeros_initializer
        if data_format == "NCHW":
            input_channel = input_shape[1]
            inputs = tf.transpose(inputs, [0, 2, 3, 1])
        elif data_format == "NHWC":
            input_channel = input_shape[3]
        else:
            raise NotImplementedError

        weights = _variable_with_weight_decay('weights',
                                              kernel_size + [input_channel, filters],
                                              wd, kerner_initializer)

        biases = _variable_on_cpu('biases', filters, bias_initializer)
        
        atrous_conv = tf.nn.atrous_conv2d(inputs, weights, rate, padding)

        if data_format == "NCHW":
            atrous_conv = tf.transpose(inputs, [0, 3, 1, 2])

        atrous_conv = tf.nn.bias_add(atrous_conv, biases, data_format=data_format)

        if bn:
            axis = -1
            if data_format == "NCWH":
                axis = 1

            atrous_conv = batch_norm_layer(atrous_conv, axis, is_train, True)

        if leaky_params is not None:
            atrous_conv = add_leaky_relu(atrous_conv, leaky_params)


    return atrous_conv


def one_hot_accuracy(infer, label, layer_name):
    """
    Args:
        infer: one hot representation
        label: one hot representation
    """
    with tf.variable_scope(layer_name):
        infer = tf.arg_max(infer, 1)
        label = tf.arg_max(label, 1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(label, infer), tf.float32))

    return accuracy


def dense_layer(x, dense_count, dense_concat_dim, layer_name,
                dense_module, *module_params, **kw_module_params):
    """
    Args:
        x: input
        dense_count: dense connected count.
        dense_concat_dim: concatenate dimention.
        layer_name: layer_name for the dense net.
        dense_module: layer for the densenet.
        *module_params: the params goes into dense_module.
        **kw_module_params: the key word params goes into dense_module
    """
    x_list = []
    with tf.variable_scope(layer_name):
        for i in range(dense_count):
            with tf.variable_scope(layer_name + "_%d" % i):
                y = dense_module(x, *module_params, **kw_module_params)
                x_list.append(y)
                x = tf.concat(x_list, dense_concat_dim)
    return x


def dropout_layer(input_tensor, dropout_rate, is_train):
    if is_train:
        dropout = tf.nn.dropout(input_tensor, dropout_rate)
    else:
        dropout = input_tensor

    return dropout
