import tensorflow as tf

class EncoderCNN(tf.keras.Model):
    def __init__(self, embed_size, num_recipes, dropout=0.5):
        self.resnet = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
        self.conv2D = tf.keras.layers.Conv2D(embed_size, 1)
        self.drop2D = tf.keras.layers.SpatialDropout2D(dropout)

    def call(self, images, keep_cnn_gradients=False):
        if keep_cnn_gradients:
            raw_conv_feats = self.resnet(images)
        else:
            raw_conv_feats = tf.stop_gradient(self.resnet(images))
        features = self.drop2D(self.conv2D(raw_conv_feats))
        features = tf.reshape(features, [features.size[0], -1, features.size[1]])
        return features