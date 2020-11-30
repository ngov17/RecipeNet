import numpy as np
import tensorflow as tf
import transformer_decoder_block as transformer
from recipe_encoder import EncoderCNN
from preprocess import get_data, classes_path, ingredients_path, images, train_image_path, test_image_path

WINDOW_SZ = 20
PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
EOS_INDEX = 2


class Ingredient_Decoder(tf.keras.Model):
    def __init__(self, ing_vocab_size):
        super(Ingredient_Decoder, self).__init__()

        # TODO:
        # 1) Define any hyperparameters
        # 2) Define embeddings, encoder, decoder, and feed forward layers
        self.vocab_size = ing_vocab_size

        # Define batch size and optimizer/learning rate
        self.batch_size = 100
        self.embedding_size = 512
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Define embedding layers:
        self.E = \
            tf.Variable(tf.random.truncated_normal([self.vocab_size, self.embedding_size], stddev=.1))

        # Create positional encoder layers
        self.image_encoder = EncoderCNN(self.embedding_size)
        self.ing_decoder = transformer.Transformer_Block(self.embedding_size, True)
        self.dense1 = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

    def call(self, images, ingredients):
        """
        :param encoder_input: batched ids corresponding to french sentences
        :param decoder_input: batched ids corresponding to english sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
        """
        pred_ings = np.zeros((ingredients.shape[0], ingredients.shape[1]), dtype='int32')
        pred_ings_inp = np.ones((ingredients.shape[0], ingredients.shape[1]), dtype='int32')
        pred_ings_inp[:, 0] = ingredients[:, 0]
        preds = []
        # ensure no repetition:
        predicted_mask = np.zeros((self.batch_size, WINDOW_SZ, self.vocab_size))
        img_features = self.image_encoder(images)
        prbs = []
        for i in range(ingredients.shape[1]):
            print(i)
            decoded_ings = self.get_decoded_ings(img_features, pred_ings_inp)
            preds.append(self.dense1(decoded_ings))
            print(preds[i].shape)   # 100 * 20 * 906
            # force preactivation of previously seen ingredients to be -inf
            if i != 0:
                for k in range(i):
                    seen_indx = pred_ings[:, k]
                    for j in range(self.batch_size):
                        predicted_mask[j][i][seen_indx[j]] = float('-inf')
            preds[i] += predicted_mask
            prbs.append(preds[i][:, i, :])
            index = tf.math.argmax(preds[i], axis=2)
            print(index.shape)
            pred_ings[:, i] = index[:, i]
            if (i + 1) < WINDOW_SZ:
                pred_ings_inp[:, i + 1] = pred_ings[:, i]
            print(pred_ings[:, i])

        return pred_ings, preds, tf.stack(prbs, axis=1)

        # Teacher Forcing:
        # img_features = self.image_encoder(images)
        # embeddings_ing = tf.nn.embedding_lookup(self.E, ingredients)
        # decoded_layer = self.ing_decoder(embeddings_ing, img_features)
        # prbs = self.dense1(decoded_layer)
        #
        # return prbs

    def get_decoded_ings(self, img_features, ingredients):

        embeddings_ing = tf.nn.embedding_lookup(self.E, ingredients)
        decoded_layer = self.ing_decoder(embeddings_ing, img_features) # 100 * 20 * 512

        return decoded_layer


    def accuracy_function(self, prbs, labels, mask):
        """

        """

        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32), mask))
        return accuracy

    def loss(self, prbs, labels, eos_indx):
        """
        Calculates the model cross-entropy loss after one forward pass
        Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

        :param prbs:  [bsz, winsz, vocabsz] or [100, 20, 906]
        :param labels: [bsz * winsz] or [100, 20]
        :return: the loss of the model as a tensor
        """
        bce = tf.keras.losses.BinaryCrossentropy()
        # EOS:
        eos_label = np.zeros_like(labels)
        eos_label[labels == eos_indx] = 1
        prb_eos = prbs[:, :, eos_indx]
        loss_eos = bce(eos_label, prb_eos)

        # Ingredient Loss:
        pooled_prbs = tf.math.reduce_max(prbs, 1)   # 100 * 906
        ingr_labels = np.zeros_like(pooled_prbs)  # 100 * 906
        for i in range(self.batch_size):
            ingr_labels[i][labels[i]] = 1
            # for j in range(WINDOW_SZ):
            #     ingr_labels[i, labels[i][j]] = 1
        loss_ingrs = bce(ingr_labels, pooled_prbs)

        return loss_eos + loss_ingrs




def main():
    print("PREPROCESSING STARTING")
    train_image, train_ingredients, test_image, test_ingredients, vocab, pad_token_idx \
        = get_data(classes_path, ingredients_path, images, train_image_path, test_image_path)
    print("PREPROCESSING DONE")

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    print(train_ingredients.shape[0])

    model = Ingredient_Decoder(len(vocab))
    train_ings = np.zeros([train_ingredients.shape[0], 20], dtype='int32')
    train_ings_label = np.zeros([train_ingredients.shape[0], 20], dtype='int32')
    for i in range(train_ingredients.shape[0]):
        train_ings[i] = train_ingredients[i][:20]
        train_ings_label[i] = train_ingredients[i][-20:]

    for j in range(0, train_ingredients.shape[0] - 100, 100):
        train_img = train_image[j:j + 100]
        train = train_ings[j:j + 100]
        labels = train_ings_label[j:j + 100]

        with tf.GradientTape() as tape:
            pred_ings, preds, prbs = model(train_img, train)
            # USE THIS VERSION IF USING TEACHER FORCING
            # prbs = model(train_img, train)
            loss = model.loss(prbs, labels, EOS_INDEX)
            print("loss at step " + str(j) + " = " + str(loss.numpy()))
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))



if __name__ == "__main__":
    main()