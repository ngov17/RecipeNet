import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import transformer_decoder_block as transformer
from recipe_encoder import EncoderCNN
from preprocess_clean import get_data, get_image_batch

data_path = os.path.join("..", "data")
classes_path = os.path.join(data_path, "classes_Recipes5k.txt")
ingredients_path = os.path.join(data_path, "ingredients_simplified_Recipes5k.txt")
images_path = os.path.join(data_path, "images")
train_image_path = os.path.join(data_path, "train_images.txt")
test_image_path = os.path.join(data_path, "test_images.txt")

WINDOW_SZ = 20
START_TOKEN = "*START*"
STOP_TOKEN = "*STOP*"
PAD_TOKEN = "*PAD*"
START_INDEX = 3
EOS_INDEX = 0
PAD_INDEX = 1
UNK_INDEX = 2


class Ingredient_Decoder(tf.keras.Model):
    def __init__(self, ing_vocab_size, window_size=20):
        super(Ingredient_Decoder, self).__init__()

        # TODO:
        # 1) Define any hyperparameters
        # 2) Define embeddings, encoder, decoder, and feed forward layers
        self.vocab_size = ing_vocab_size
        self.window_size = window_size

        # Define batch size and optimizer/learning rate
        self.batch_size = 64
        self.embedding_size = 512
        self.learning_rate = 0.0001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Define embedding layers:
        self.E = tf.Variable(tf.random.truncated_normal([self.vocab_size, self.embedding_size], stddev=.1))

        # Create positional encoder layers
        self.image_encoder = EncoderCNN(self.embedding_size)
        self.ing_decoder = transformer.Transformer_Block(self.embedding_size, True)
        self.dense1 = tf.keras.layers.Dense(self.vocab_size)

    def call(self, images, ingredients, teacher_forcing=False):
        """
        :param encoder_input: batched ids corresponding to french sentences
        :param decoder_input: batched ids corresponding to english sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
        """
        # pred_ings = np.zeros((ingredients.shape[0], ingredients.shape[1]), dtype='int32')
        # pred_ings_inp = np.ones((ingredients.shape[0], ingredients.shape[1]), dtype='int32')
        # pred_ings_inp[:, 0] = ingredients[:, 0]
        # preds = []
        # # ensure no repetition:
        # predicted_mask = np.zeros((self.batch_size, WINDOW_SZ, self.vocab_size))
        # img_features = self.image_encoder(images)
        # prbs = []
        # for i in range(WINDOW_SZ): # iterating 20 times
        #     print(i)
        #     decoded_ings = self.get_decoded_ings(img_features, pred_ings_inp)
        #     preds.append(self.dense1(decoded_ings))
        #     print(preds[i].shape)   # 100 * 1 * 906
        #     # force preactivation of previously seen ingredients to be -inf
        #     if i != 0:
        #         for k in range(i):
        #             seen_indx = pred_ings[:, k]
        #             for j in range(self.batch_size):
        #                 predicted_mask[j][i][seen_indx[j]] = float('-inf')
        #     preds[i] += predicted_mask
        #     prbs.append(preds[i][:, i, :])
        #     index = tf.math.argmax(preds[i], axis=2)
        #     print(index.shape)
        #     pred_ings[:, i] = index[:, i]
        #     if (i + 1) < WINDOW_SZ:
        #         pred_ings_inp[:, i + 1] = pred_ings[:, i]
        #     print(pred_ings[:, i])
        #
        # return pred_ings, preds, tf.stack(prbs, axis=1)

        if teacher_forcing:
            # Teacher Forcing:
            img_features = self.image_encoder(images)
            embeddings_ing = tf.nn.embedding_lookup(self.E, ingredients)
            decoded_layer = self.ing_decoder(embeddings_ing, img_features)
            prbs = self.dense1(decoded_layer)
            return prbs
        else:
            first_word = tf.ones(images.shape[0], dtype=tf.int64) * START_INDEX
            sampled_ids = [first_word]
            logits = []
            image_features = self.image_encoder(images)
            for i in range(WINDOW_SZ):
                print(i)
                # forward pass:
                output = self.get_decoded_ings(image_features, tf.stack(sampled_ids, axis=1))  # batch_size * 1 * 906
                output = tf.squeeze(output, axis=1)  # batch_size x 906
                # force preactivation of previously seen elements to -inf:
                if i == 0:
                    predicted_mask = np.zeros(output.shape, dtype='float32')
                else:
                    # ensure no repetitions in sampling if replacement==False
                    batch_ind = tf.where(tf.math.not_equal(sampled_ids[i], tf.constant(PAD_INDEX, dtype=tf.int64)))
                    batch_ind = tf.squeeze(batch_ind)
                    sampled_ids_new = tf.gather(sampled_ids[i], batch_ind)
                    predicted_mask[batch_ind.numpy(), sampled_ids_new.numpy()] = float('-inf')

                # mask previously seen ingredients:
                output += predicted_mask
                logits.append(output)

                # get the predicted ingredient from the output:
                output_prb = tf.nn.softmax(output)
                output_pred = tf.math.argmax(output_prb, axis=1)

                sampled_ids.append(output_pred)
            sampled_ids = tf.stack(sampled_ids[1:], 1)
            logits = tf.stack(logits, 1)

            return sampled_ids, logits

    def get_decoded_ings(self, img_features, ingredients):
        # since we are incrementally inputting ingredients into the model, we get the ing from the last time step:
        ingredients = ingredients[:, -1:] # batch_size * 1

        embeddings_ing = tf.nn.embedding_lookup(self.E, ingredients)
        decoded_layer = self.ing_decoder(embeddings_ing, img_features) # 100 * 1 * 512
        prbs = self.dense1(decoded_layer) # 100 * 1 * 906

        return prbs

    def accuracy_function(self, prbs, labels, mask):
        """

        """

        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32), mask))
        return accuracy

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass
        Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

        :param prbs:  [bsz, winsz, vocabsz] or [100, 20, 906]
        :param labels: [bsz * winsz] or [100, 20]
        :return: the loss of the model as a tensor
        """
        prbs = tf.nn.softmax(logits)
        bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        # EOS:
        eos_label = np.zeros_like(labels)
        eos_label[(labels == EOS_INDEX) ^ (labels == PAD_INDEX)] = 1
        eos_pos = np.zeros_like(labels)
        eos_pos[(labels == EOS_INDEX)] = 1
        eos_head = np.zeros_like(labels)
        eos_head[(labels != PAD_INDEX) & (labels != EOS_INDEX)] = 1
        prb_eos = prbs[:, :, EOS_INDEX]
        loss_eos = tf.reduce_sum(bce(eos_pos, prb_eos))
        print(loss_eos)
        # Ingredient Loss:
        pooled_prbs = tf.math.reduce_max(prbs, 1)   # 100 * 906
        ingr_labels = np.zeros_like(pooled_prbs)  # 100 * 906
        for i in range(self.batch_size):
            ingr_labels[i][labels[i]] = 1
            ingr_labels[i][PAD_INDEX] = 0
            ingr_labels[i][EOS_INDEX] = 0
            # for j in range(WINDOW_SZ):
            #     if (labels[i][j] != PAD_INDEX) & (labels[i][j] != EOS_INDEX):
            #         ingr_labels[i, labels[i][j]] = 1

        loss_ingrs = bce(ingr_labels, pooled_prbs)
        print(loss_ingrs.shape)
        loss_ingrs = tf.reduce_sum(loss_ingrs)
        print(loss_ingrs)

        return loss_eos + loss_ingrs

def train(model, train_image_paths, train_ings, train_ings_labels, num_epochs=1, shuffle=True):
    for n in range(num_epochs):
        print("Epoch " + str(n))
        # shuffle inputs:
        indices = np.arange(train_ings.shape[0])
        if shuffle:
            np.random.shuffle(indices)
        for j in range(0, train_ings.shape[0] - model.batch_size, model.batch_size):
            batch_indices = indices[j:j + model.batch_size]
            batch_train_image_paths = [train_image_paths[idx] for idx in batch_indices] # Have to do it like this because train_image is a list of strings now and can't use numpy's array indexing
            train_img = get_image_batch(batch_train_image_paths, is_train=True)
            train_ing = train_ings[batch_indices]
            labels = train_ings_labels[batch_indices]
            with tf.GradientTape() as tape:
                # sampled_ids, logits = model(train_img, train_ing)
                # USE THIS VERSION IF USING TEACHER FORCING
                logits = model(train_img, train_ing, teacher_forcing=True)
                loss = model.loss(logits, labels)
                print("loss at step" + str(j) + " = " + str(loss.numpy()))
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def main():
    print("PREPROCESSING STARTING")
    train_image_paths, train_ingredients, test_image_paths, test_ingredients, vocab \
        = get_data(classes_path, ingredients_path, images_path, train_image_path, test_image_path)
    print("PREPROCESSING DONE")

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    print(train_ingredients.shape[0])

    model = Ingredient_Decoder(len(vocab), WINDOW_SZ)
    train_ings = train_ingredients[:,0:20]
    train_ings_label = train_ingredients[:,1:21]
    train(model, train_image_paths, train_ings, train_ings_label, num_epochs=1)
    # model is trained
    im_0 = get_image_batch([train_image_paths[0]], is_train=False)
    im_1 = get_image_batch([train_image_paths[127]], is_train=False)
    print(f"im_0 shape: {im_0.shape}") # Should be (1, 224, 224, 3)
    start = [[START_INDEX]]
    ing_id_list_0 = model(im_0, start)[0][0] # len 20 list of ingredient IDs
    ing_id_list_1 = model(im_1, start)[0][0] # len 20 list of ingredient IDs

    def ingredient_ids_to_strings(id_list):
        return [reverse_vocab[ing_id.numpy()] for ing_id in id_list]

    print("Image 0 prediction: ")
    print(ingredient_ids_to_strings(ing_id_list_0))
    plt.imshow(im_0[0])
    plt.show()

    print("Image 1 prediction: ")
    print(ingredient_ids_to_strings(ing_id_list_1))
    plt.imshow(im_1[0])
    plt.show()

if __name__ == "__main__":
    main()
