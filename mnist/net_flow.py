import cv2
import numpy as np
import os
import tensorflow as tf

import file_io
import save_func as sf
import utility_func as uf
import mnist_data_input
import mnist_data_ph


class NetFlow(object):
    def __init__(self, model_params, load_train, load_test):
        load_dapt = model_params['adapt']

        self.load_train = load_train
        self.load_dapt = load_dapt
        self.load_test = load_test
        self.model_params = model_params

        if load_train:
            self.source_data_input = list()
            for i in range(model_params["source_num"]):
                self.source_data_input.append(
                    mnist_data_input.DataInput(model_params,
                                               model_params['source_type%d' % (
                                                   i + 1)], 'source%d' % (i + 1),
                                               is_train=True))

        if load_dapt:
            self.target_data_input = list()
            for i in range(model_params["target_num"]):
                self.target_data_input.append(
                    mnist_data_input.DataInput(model_params,
                                               model_params['target_type%d' % (
                                                   i + 1)], 'target%d' % (i + 1),
                                               is_train=True))

        if load_test:
            self.test_data_input = mnist_data_input.DataInput(model_params,
                                                              model_params['target_type1'], 'test',
                                                              is_train=False)
            if not load_dapt:
                self.target_data_input = list()
                self.target_data_input.append(self.test_data_input)

        self.data_ph = mnist_data_ph.DataPh(model_params)
        model = file_io.import_module_class(model_params["model_def_name"],
                                            "Model")

        self.model = model(self.data_ph, model_params)
        self.loss = self.model.get_loss()
        self.accuracy = self.model.get_accuracy()
        self.train_op = self.model.get_train_op()

    def get_feed_dict(self, sess, is_train):
        """
        Args:
            is_train: True or False
        """

        feed_dict = dict()

        input_v = list()
        label_v = list()

        if is_train:
            for i in range(self.model_params["source_num"]):
                input_vi, label_vi = self.source_data_input[i].load_data()
                # input_v.append(sess.run(input_vi))
                input_v.append(input_vi)
                label_v.append(label_vi)

            if self.model_params['adapt']:
                for i in range(self.model_params["target_num"]):
                    dapt_input_vi, dapt_label_vi = self.target_data_input[i].load_data(
                    )
                    # input_v.append(sess.run(dapt_input_vi))
                    input_v.append(dapt_input_vi)
                    label_v.append(dapt_label_vi)

            feed_dict[self.data_ph.get_keep_prob()] = 0.5
            feed_dict[self.data_ph.get_train_test()] = True
        else:
            input_vi, label_vi = self.test_data_input.load_data()

            # input_v.append(sess.run(input_vi))
            input_v.append(input_vi)
            label_v.append(label_vi)

            # if testing, the daptive input will be the same test queue
            if self.model_params['adapt']:
                input_v *= self.model_params["source_num"] + \
                    self.model_params["target_num"]
                label_v *= self.model_params["source_num"] + \
                    self.model_params["target_num"]
            else:
                input_v *= self.model_params["source_num"]
                label_v *= self.model_params["source_num"]

            feed_dict[self.data_ph.get_keep_prob()] = 1.0
            feed_dict[self.data_ph.get_train_test()] = False

        input_v = np.concatenate(input_v, axis=0)
        label_v = np.concatenate(label_v, axis=0)

        feed_dict[self.data_ph.get_input()] = input_v
        feed_dict[self.data_ph.get_label()] = label_v
        return feed_dict

    def init_var(self, sess):
        sf.add_train_var()
        sf.add_loss()
        sf.add_image("image_to_write", 4)
        self.saver = tf.train.Saver()

        if self.load_train:
            self.sum_writer = tf.summary.FileWriter(self.model_params["train_log_dir"],
                                                    sess.graph)
        self.summ = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()

        sess.run(init_op, feed_dict={self.data_ph.get_train_test(): True})

        if self.model_params["restore_model"]:
            sf.restore_model(sess, self.saver, self.model_params["model_dir"],
                             self.model_params["restore_model_name"])

    def mainloop(self):
        config_proto = uf.define_graph_config(self.model_params["gpu_fraction"])
        sess = tf.Session(config=config_proto)
        self.init_var(sess)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        if self.load_train:
            for i in range(self.model_params["max_training_iter"]):

                feed_dict = self.get_feed_dict(sess, True)
                p = float(i) / self.model_params["max_training_iter"]
                l = 2. / (1. + np.exp(-10. * p)) - 1
                feed_dict[self.data_ph.get_gl()] = \
                    l * self.model_params['adapt_scale']

                _, tloss_v, taccuracy_v = sess.run([self.train_op,
                                                    self.loss, self.accuracy], feed_dict)

                if i % self.model_params["test_per_iter"] == 0:
                    feed_dict = self.get_feed_dict(sess, False)
                    loss_v, accuracy_v, summ_v = \
                        sess.run([self.loss, self.accuracy,
                                  self.summ], feed_dict)

                    print_string = "i: %d, train_loss: %.2f, "\
                        "train_accuracy: %.2f, "\
                        "test_loss: %.2f, test_accuracy: %.2f" %\
                        (i, tloss_v, taccuracy_v, loss_v, accuracy_v)
                    print(print_string)

                    file_io.save_string(print_string,
                                        self.model_params["train_log_dir"] +
                                        self.model_params["string_log_name"])

                    self.sum_writer.add_summary(summ_v, i)
                    sf.add_value_sum(self.sum_writer, tloss_v, "train_loss", i)
                    sf.add_value_sum(self.sum_writer, loss_v, "test_loss", i)
                    sf.add_value_sum(
                        self.sum_writer, taccuracy_v, "train_accu", i)
                    sf.add_value_sum(
                        self.sum_writer, accuracy_v, "test_accu", i)

                if i != 0 and (i % self.model_params["save_per_iter"] == 0 or
                               i == self.model_params["max_training_iter"] - 1):
                    sf.save_model(sess, self.saver,
                                  self.model_params["model_dir"], i)

        else:
            file_len = self.target_data_input[0].file_size
            batch_size = self.model_params["batch_size"]
            test_iter = int(file_len / batch_size) + 1
            accuracy_list = list()
            for i in range(test_iter):
                feed_dict = self.get_feed_dict(sess, False)
                loss_v, accuracy_v = sess.run(
                    [self.loss, self.accuracy], feed_dict)
                accuracy_list.append(accuracy_v)

                print("accuracy is %.2f" % accuracy_v)

            accuracy = np.mean(np.array(accuracy_list))
            accuracy_string = "final accuracy is: %.3f" % accuracy
            print(accuracy_string)
            with open(self.model_params["result_file_name"], 'w') as f:
                f.write(accuracy_string)

        coord.request_stop()
        coord.join(threads)
