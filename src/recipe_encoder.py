import tensorflow as tf
from preprocess import get_data, classes_path, ingredients_path, images, train_image_path, test_image_path


class EncoderCNN(tf.keras.Model):
    def __init__(self, embed_size, dropout=0.5):
        super(EncoderCNN, self).__init__()
        self.resnet = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
        self.conv2D = tf.keras.layers.Conv2D(embed_size, 1)
        self.drop2D = tf.keras.layers.SpatialDropout2D(dropout)

    def call(self, images, keep_cnn_gradients=False):
        if keep_cnn_gradients:
            raw_conv_feats = self.resnet(images)
        else:
            raw_conv_feats = tf.stop_gradient(self.resnet(images))
        features = self.drop2D(self.conv2D(raw_conv_feats))
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

