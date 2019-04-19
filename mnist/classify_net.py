import model_func as mf
import tensorflow as tf

from flip_gradient import flip_gradient


class Model(object):
    def __init__(self, data_ph, model_params):
        self.model_params = model_params
        self.model_infer(data_ph, model_params)
        self.model_loss(data_ph, model_params)
        self.model_mini(model_params)

    def add_data_arg(self, data):
        data = tf.map_fn(
            lambda img: tf.image.random_brightness(img, 0.2),
            data)

        data = tf.map_fn(lambda img:
                         tf.image.random_contrast(img, 0.5, 1.5),
                         data)
        return data

    def model_infer(self, data_ph, model_params):
        input_ph = data_ph.get_input()
        gl_ph = data_ph.get_gl()
        train_test_ph = data_ph.get_train_test()

        input_ph = tf.cond(train_test_ph, lambda: self.add_data_arg(
            input_ph), lambda: input_ph)

        data_format = "NHWC"
        bn = False
        keep_prob_ph = data_ph.get_keep_prob()
        self.keep_prob_ph = keep_prob_ph

        b, _, _, _ = input_ph.get_shape()
        self.b = b

        leaky_param = model_params["leaky_param"]
        wd = model_params["weight_decay"]

        hyper_list = list()

        print(input_ph)
        conv11 = mf.convolution_2d_layer(
            input_ph, 64, [3, 3], [1, 1], "SAME", data_format=data_format,
            leaky_params=leaky_param, wd=wd, layer_name="conv11")
        print(conv11)

        conv1_maxpool = mf.maxpool_2d_layer(conv11, [2, 2],
                                            [2, 2], data_format, "maxpool1")
        print(conv1_maxpool)

        conv21 = mf.convolution_2d_layer(
            conv1_maxpool, 128, [3, 3], [1, 1], "SAME", data_format=data_format,
            leaky_params=leaky_param, wd=wd, layer_name="conv21")
        print(conv21)

        conv2_maxpool = mf.maxpool_2d_layer(conv21, [2, 2],
                                            [2, 2], data_format, "maxpool2")
        print(conv2_maxpool)

        conv31 = mf.convolution_2d_layer(
            conv2_maxpool, 256, [3, 3], [1, 1], "SAME", data_format=data_format,
            leaky_params=leaky_param, wd=wd, layer_name="conv31")

        print(conv31)

        if model_params['adapt']:
            if self.model_params["adapt_loss_type"] == "MULTI":
                self._add_adapt_multi_loss(conv31,
                                           gl_ph, leaky_param, wd)
            elif self.model_params["adapt_loss_type"] == "PAIR":
                self._add_adapt_pair_loss(conv31,
                                          gl_ph, leaky_param, wd)
            else:
                raise NotImplementedError

        conv3_maxpool = mf.maxpool_2d_layer(
                conv31, [2, 2], [2, 2], data_format, "maxpool3")

        conv41 = mf.convolution_2d_layer(
            conv3_maxpool, 256, [3, 3], [1, 1], "SAME", data_format=data_format,
            leaky_params=leaky_param, wd=wd, layer_name="conv4")

        fc1 = mf.fully_connected_layer(
            conv41, 2048, leaky_param, wd, "fc1")

        fc1_drop = tf.nn.dropout(fc1, keep_prob_ph, name="dropout1")

        fc2 = mf.fully_connected_layer(
            fc1_drop, 1024, leaky_param, wd, "fc2")

        fc2_drop = tf.nn.dropout(fc2, keep_prob_ph, name="dropout2")

        fc3 = mf.fully_connected_layer(fc2_drop, 10, 0.0, wd, "fc3")

        self.fc = fc3

    def _add_adapt_multi_loss(self, feature, scale, leaky_param, wd):
        raise NotImplementedError

    def _add_adapt_pair_loss_single(self, feature, scale, leaky_param,
                                    wd, layer_name):
        with tf.variable_scope(layer_name):
            feat = flip_gradient(feature, scale)
            keep_prob_ph = self.keep_prob_ph
            data_format = "NHWC"

            conv1 = mf.convolution_2d_layer(
                feat, 256, [3, 3], [1, 1],
                "SAME", data_format=data_format, leaky_params=leaky_param,
                wd=wd, layer_name="dann_conv1")

            conv2 = mf.convolution_2d_layer(
                conv1, 512, [3, 3], [1, 1],
                "SAME", data_format=data_format, leaky_params=leaky_param,
                wd=wd, layer_name="dann_conv2")

            conv3 = mf.convolution_2d_layer(
                conv2, 512, [3, 3], [1, 1],
                "SAME", data_format=data_format, leaky_params=leaky_param,
                wd=wd, layer_name="dann_conv3")

            conv3_maxpool = mf.maxpool_2d_layer(
                conv3, [2, 2], [2, 2], data_format=data_format,
                layer_name="dann_maxpool3")

            a_fc1 = mf.fully_connected_layer(
                conv3_maxpool, 2048, leaky_param, wd, "a_fc1")

            a_fc2 = mf.fully_connected_layer(
                a_fc1, 2048, leaky_param, wd, "a_fc2")

            a_fc3 = mf.fully_connected_layer(
                a_fc2, 2, leaky_param, wd, "a_fc3")

        return a_fc3

    def _add_adapt_pair_loss(self, feature, scale, leaky_param, wd):
        source_num = self.model_params["source_num"]
        target_num = self.model_params["target_num"]
        source_target_num = source_num + target_num

        b = self.b

        single_b = tf.div(b, source_target_num)
        label_shape = tf.convert_to_tensor([single_b, 1])

        source_label = tf.tile(tf.one_hot([0], 2),
                               label_shape)
        target_label = tf.tile(tf.one_hot([1], 2),
                               label_shape)
        class_label = tf.concat([source_label, target_label], 0)

        for i in range(target_num):
            target_tensor = feature[single_b * (i + source_num):
                                    single_b * (i + source_num + 1)]
            pair_loss_list = list()
            for j in range(source_num):
                source_tensor = feature[single_b * j:
                                        single_b * (j + 1)]

                curr_pair = tf.concat([source_tensor, target_tensor], 0)
                fc = self._add_adapt_pair_loss_single(curr_pair, scale,
                                                      leaky_param, wd, "adapt_%d_%d" % (i, j))

                pair_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=fc, labels=class_label),
                    name='pair_loss_%d_%d' % (i, j))
                pair_loss_list.append(pair_loss)

            if self.model_params["dann_loss"] == "MEAN":
                pair_loss = tf.reduce_mean(pair_loss_list)

            elif self.model_params["dann_loss"] == "MIN":
                pair_loss = tf.reduce_min(pair_loss_list)

            elif self.model_params["dann_loss"] == "MAX":
                pair_loss = tf.reduce_max(pair_loss_list)

            elif self.model_params["dann_loss"] == "SOFTMIN":
                negative_loss_list = list()
                for l in pair_loss_list:
                    negative_loss_list.append(
                        tf.negative(l))
                weight = flip_gradient(
                    tf.nn.softmax(
                        tf.convert_to_tensor(
                            negative_loss_list)), 0)
                loss_tensor = tf.convert_to_tensor(pair_loss_list)
                pair_loss = tf.reduce_sum(tf.multiply(weight, loss_tensor))
            else:
                raise NotImplementedError

            tf.add_to_collection("losses", pair_loss)

    def _get_source(self, tensor):
        b = tensor.get_shape()[0]
        source_num = self.model_params["source_num"]
        target_num = self.model_params["target_num"]
        source_target_num = source_num + target_num
        single_b = b / source_target_num
        source_b = single_b * source_num
        tensor = tensor[:source_b]
        return tensor

    def _get_xentropy_loss(self, fc, label, loss_list_type):
        source_num = self.model_params["source_num"]
        b = fc.get_shape().as_list()[0]
        single_b = int(b / source_num)
        batch_x_loss_list = list()

        with tf.variable_scope("select_l2_loss"):
            for i in range(source_num):
                curr_fc = fc[single_b * i: single_b * (i + 1)]
                curr_label = label[single_b * i: single_b * (i + 1)]

                batch_x_loss = mf.x_entropy_loss(
                    curr_fc, curr_label, "x_loss_%d" % i)
                batch_x_loss_list.append(batch_x_loss)

            if loss_list_type == "MAX":
                x_loss = tf.reduce_max(batch_x_loss_list)
            elif loss_list_type == "MIN":
                x_loss = tf.reduce_min(batch_x_loss_list)
            elif loss_list_type == "MEAN":
                x_loss = tf.reduce_mean(batch_x_loss_list)
            elif loss_list_type == "SOFTMAX":
                weight = flip_gradient(tf.nn.softmax(batch_x_loss_list), 0)

                x_loss = tf.reduce_sum(tf.multiply(
                    weight,
                    batch_x_loss_list))
            else:
                raise NotImplementedError

        return x_loss

    def model_loss(self, data_ph, model_params):
        with tf.variable_scope("loss"):
            label = data_ph.get_label()
            label = tf.one_hot(label, 10)

            if model_params['adapt']:
                fc = self._get_source(self.fc)
                label = self._get_source(label)
            else:
                fc = self.fc

            x_loss = self._get_xentropy_loss(fc, label,
                                             self.model_params["class_loss_type"])

            self.accuracy = mf.one_hot_accuracy(fc, label, "accuracy")

            tf.add_to_collection("losses", x_loss)

            self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    def model_mini(self, model_params):
        with tf.variable_scope("optimization"):
            optimizer = tf.train.AdamOptimizer(
                model_params["init_learning_rate"],
                epsilon=1.0)
            self.train_op = optimizer.minimize(self.loss)

    def get_train_op(self):
        return self.train_op

    def get_accuracy(self):
        return self.accuracy

    def get_loss(self):
        return self.loss

    def get_l2_loss(self):
        pass

    def get_l1_loss(self):
        pass

    def get_count_diff(self):
        pass

    def get_count(self):
        pass

    def get_label_count(self):
        pass


if __name__ == "__main__":
    pass
