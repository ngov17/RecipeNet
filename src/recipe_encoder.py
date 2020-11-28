import tensorflow as tf

class EncoderCNN(tf.keras.Model):
    def __init__(self, embed_size, num_recipes, dropout=0.5):
        self.resnet = tf.keras.Sequential()
        self.linear = tf.keras.Sequential()
        self.resnet.add(tf.keras.applications.ResNet50(include_top=False, classes=num_recipes).output)
        self.linear.add(tf.keras.layers.Conv2D(embed_size, 1))
        self.linear.add(tf.keras.layers.SpatialDropout2D(dropout))

    def call(self, images, keep_cnn_gradients=False):
        if keep_cnn_gradients:
            raw_conv_feats = self.resnet(images)
        else:
            raw_conv_feats = tf.stop_gradient(self.resnet(images)) #NOTE: this may not work
        features = self.linear(raw_conv_feats)
        features = tf.reshape(features, [features.size[0], -1, features.size[1]])
        return features