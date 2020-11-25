import tensorflow as tf

class EncoderCNN(tf.keras.Model): #TODO: is nn.Module equivalent to keras.Model, or should this be a Layer?
    def __init__(self, embed_size, num_recipes, dropout=0.5):
        resnet = tf.keras.applications.ResNet50(classes=num_recipes)
        modules = resnet.layers[-1].output #TODO: change the name to match tensorflow instead of pytorch
        self.resnet = tf.keras.Sequential(modules) #TODO: make sure keras.Sequential is equivalent to torch.nn.Sequential
        self.linear = tf.keras.Sequential(tf.keras.layers.Conv2D(embed_size, 1), tf.keras.layers.SpatialDropout2D(dropout))
        #torch.nn.Conv2d(in_channels: int, out_channels: int, kernel_size=1, padding=0)
        #tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')
    def call(self, images, keep_cnn_gradients=False):
        if keep_cnn_gradients:
            raw_conv_feats = self.resnet(images)
        else:
            raw_conv_feats = tf.stop_gradient(self.resnet(images))
        features = self.linear(raw_conv_feats)
        features = tf.reshape(features, [features.size[0], -1, features.size[1]])
        return features