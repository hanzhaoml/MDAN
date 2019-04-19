import tensorflow as tf


class DataPh(object):
    def __init__(self, model_params):
        batch_size = model_params['batch_size']
        if model_params['adapt']:
            batch_size *= model_params["source_num"] + \
                model_params["target_num"]
        else:
            batch_size *= model_params["source_num"]

        self.input_ph = tf.placeholder(tf.float32, shape=[
            batch_size,
            model_params["feature_ph_row"],
            model_params["feature_ph_col"],
            model_params["feature_cha"]],
            name="feature")

        self.label_ph = tf.placeholder(tf.uint8, shape=[
            batch_size], name="label")

        self.train_test_ph = tf.placeholder(tf.bool,
                                            name="train_test")

        self.keep_prob_ph = tf.placeholder(tf.float32,
                                           name="keep_prob")

        self.gl = tf.placeholder(tf.float32,
                                 name="lambda")

    def add_data_arg(self):
        self.input_ph = tf.map_fn(
            lambda img: tf.image.random_brightness(img, 0.2),
            self.input_ph_org)

        self.input_ph = tf.map_fn(lambda img:
                                  tf.image.random_contrast(img, 0.5, 1.5),
                                  self.input_ph_org)

    def get_label(self):
        return self.label_ph

    def get_input(self):
        return self.input_ph

    def get_input_org(self):
        return self.input_ph

    def get_keep_prob(self):
        return self.keep_prob_ph

    def get_gl(self):
        return self.gl

    def get_train_test(self):
        return self.train_test_ph
