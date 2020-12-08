import tensorflow as tf
from preprocess import get_data, classes_path, ingredients_path, images, train_image_path, test_image_path


class EncoderCNN(tf.keras.Model):
    def __init__(self, embed_size, dropout=0.3):
        super(EncoderCNN, self).__init__()

        # hyper parameters
        self.h1 = 256

        # layers
        self.resnet = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        self.resnet.trainable = False
        self.conv2D = tf.keras.layers.Conv2D(self.h1, 1, activation='relu', padding='SAME')
        self.conv2D1 = tf.keras.layers.Conv2D(embed_size, 1, padding='SAME')
        self.drop2D = tf.keras.layers.SpatialDropout2D(dropout)

    def call(self, images, keep_cnn_gradients=False):
        if keep_cnn_gradients:
            raw_conv_feats = self.resnet(images)
        else:
            raw_conv_feats = tf.stop_gradient(self.resnet(images))
        conv2D_output = self.conv2D(raw_conv_feats)
        mean1, variance1 = tf.nn.moments(conv2D_output, [0, 1, 2])
        conv2D_output = tf.nn.batch_normalization(conv2D_output, mean1, variance1,
                                                  offset=None, scale=None, variance_epsilon=1e-5)
        features = self.conv2D1(conv2D_output)
        mean2, variance2 = tf.nn.moments(features, [0, 1, 2])
        features = tf.nn.batch_normalization(features, mean2, variance2,
                                                  offset=None, scale=None, variance_epsilon=1e-5)

        features = self.drop2D(features)
        features = tf.reshape(features, [features.shape[0], -1, features.shape[-1]])

        return features


def main():
    train_image, train_ingredients, test_image, test_ingredients, vocab, pad_token_idx \
        = get_data(classes_path, ingredients_path, images, train_image_path, test_image_path)

    train = train_image[:100]

    model = EncoderCNN(512)

    features = model(train)

    print(features.shape)


if __name__ == "__main__":
    main()
