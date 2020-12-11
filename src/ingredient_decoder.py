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
START_INDEX = 2  # will be the same as UNK index as START token is not in vocab
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
        self.window_size = WINDOW_SZ

        # Define batch size and optimizer/learning rate

        self.batch_size = 64
        self.img_embedding_size = 512
        self.embedding_size = 512
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        # Define embedding amd hidden layers:
        self.E = \
            tf.Variable(tf.random.truncated_normal([self.vocab_size, self.embedding_size], stddev=.1))
        self.h1 = self.vocab_size

        # Layers
        self.image_encoder = EncoderCNN(self.img_embedding_size)
        self.ing_decoder = transformer.Transformer_Block(self.embedding_size, True, multi_headed=True)
        self.ing_decoder1 = transformer.Transformer_Block(self.embedding_size, True, multi_headed=True)
        self.ing_decoder2 = transformer.Transformer_Block(self.embedding_size, True, multi_headed=True)
        self.ing_decoder3 = transformer.Transformer_Block(self.embedding_size, True, multi_headed=True)
        self.dense1 = tf.keras.layers.Dense(self.vocab_size)

    def call(self, images, ingredients, teacher_forcing=False, sample_k=None):
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
        if sample_k is not None:
            assert teacher_forcing is False

        if teacher_forcing:
            # Teacher Forcing:
            img_features = self.image_encoder(images)
            embeddings_ing = tf.nn.embedding_lookup(self.E, ingredients)
            decoded_layer = self.ing_decoder(embeddings_ing, img_features)  # bsz * 20 * 512
            decoded_layer1 = self.ing_decoder1(decoded_layer, img_features)
            decoded_layer2 = self.ing_decoder2(decoded_layer1, img_features)
            decoded_layer3 = self.ing_decoder3(decoded_layer2, img_features)
            logits = self.dense1(decoded_layer3)

            return None, logits
        else:
            # ingredients[i][0] for any i is START token idx
            first_word = tf.ones(images.shape[0], dtype=tf.int64) * ingredients[0][0]
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

                if sample_k is not None:
                    top_k, indices_k = tf.math.top_k(output, k=sample_k)
                    # # choose random number between 0 and 10
                    # rand = np.random.randint(low=0, high=10)
                    # output_pred = indices_prbs[:, rand]
                    # output_pred = tf.cast(output_pred, dtype=tf.int64)
                    # top_k = np.argsort(output, axis=1)[:, -sample_k:]
                    output_pred = tf.random.categorical(top_k, 1)
                    output_pred = tf.reshape(output_pred, -1)
                    output_pred = tf.gather(indices_k, output_pred, batch_dims=1)
                    output_pred = tf.cast(output_pred, dtype=tf.int64)


                else:
                    # get the predicted ingredient from the output:
                    output_prb = tf.nn.softmax(output)
                    output_pred = tf.math.argmax(output_prb, axis=1)

                sampled_ids.append(output_pred)
            sampled_ids = tf.stack(sampled_ids[1:], 1)
            logits = tf.stack(logits, 1)

            return sampled_ids, logits

    def get_decoded_ings(self, img_features, ingredients):
        # since we are incrementally inputting ingredients into the model, we get the ing from the last time step:
        ingredients = ingredients[:, -1:]  # batch_size * 1

        embeddings_ing = tf.nn.embedding_lookup(self.E, ingredients)  #
        decoded_layer = self.ing_decoder(embeddings_ing, img_features)  # bsz * 1 * 512
        decoded_layer1 = self.ing_decoder1(decoded_layer, img_features)
        decoded_layer2 = self.ing_decoder2(decoded_layer1, img_features)
        decoded_layer3 = self.ing_decoder3(decoded_layer2, img_features)
        logits = self.dense1(decoded_layer3)

        return logits

    def accuracy_function(self, prbs, labels, mask):
        """
        """
        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32), mask))
        return accuracy

    def label2onehot(self, labels, vocab_sz):
        """
        labels: batch_size x window_size x vocab
        outputs: batch_size x window_size x vocab size, where vocab size is onehot
        """
        one_hot_labels = np.zeros([labels.shape[0], labels.shape[1], vocab_sz], dtype=np.int64)
        for i in range(labels.shape[0]):
            one_hot = tf.one_hot(labels[i], vocab_sz)
            one_hot_labels[i] = one_hot

        return one_hot_labels

    def loss(self, logits, labels, mask, teacher_forcing=False):
        """
        Calculates the model cross-entropy loss after one forward pass
        Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

        :param prbs:  [bsz, winsz, vocabsz] or [100, 20, 906]
        :param labels: [bsz * winsz] or [100, 20]
        :return: the loss of the model as a tensor
        """



        # weights for each loss
        loss_eos_w = 0.2
        loss_ingr_w = 0.8

        prbs = tf.nn.softmax(logits)
        bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        if teacher_forcing:
            # for teacher forcing, we use sparse categorical cross entropy
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs)
            masked_loss = tf.boolean_mask(loss, mask)
            return tf.reduce_sum(masked_loss)

        # EOS Loss:
        eos_label = np.zeros_like(labels)
        eos_label[(labels == EOS_INDEX) ^ (labels == PAD_INDEX)] = 1
        eos_pos = np.zeros_like(labels, dtype='float32')
        eos_pos[(labels == EOS_INDEX)] = 1
        eos_head = np.zeros_like(labels, dtype='float32')
        eos_head[(labels != PAD_INDEX) & (labels != EOS_INDEX)] = 1
        prb_eos = prbs[:, :, EOS_INDEX]
        loss_eos = bce(eos_label, prb_eos)
        loss_eos = tf.reshape(loss_eos, (loss_eos.shape[0], 1))
        loss_eos = 0.5 * tf.reduce_sum((loss_eos * eos_pos), axis=1) / (tf.reduce_sum(eos_pos, axis=1) + 1e-6) + \
                   0.5 * tf.reduce_sum((loss_eos * eos_head), axis=1) / (tf.reduce_sum(eos_head, axis=1) + 1e-6)
        loss_eos = tf.reduce_mean(loss_eos)
        print(loss_eos)
        # Ingredient Loss:
        # prbs = tf.boolean_mask(prbs, mask, axis=1)
        pooled_prbs = tf.math.reduce_max(prbs, 1)  # bsz * 906
        mask_pooled = np.ones_like(pooled_prbs)
        mask_pooled[:, EOS_INDEX] = 0
        mask_pooled[:, PAD_INDEX] = 0
        pooled_prbs = tf.math.multiply(pooled_prbs, mask_pooled)

        # pooled_prbs, indices_prbs = tf.math.top_k(pooled_prbs, k=20)

        ingr_labels = np.zeros_like(pooled_prbs)  # 100 * 906
        # ingr_labels = self.label2onehot(labels, self.vocab_size) # bsz x wsz x vocabsize
        # set_ingr_loss = bce(ingr_labels, prbs)
        # set_ingr_loss = tf.reduce_sum(set_ingr_loss)
        # print(set_ingr_loss)
        for i in range(self.batch_size):
            ingr_labels[i][labels[i]] = 1
            # for j in range(WINDOW_SZ):
            #     if (labels[i][j] != PAD_INDEX) & (labels[i][j] != EOS_INDEX):
            #         ingr_labels[i, labels[i][j]] = 1
        loss_ingrs1 = bce(ingr_labels, pooled_prbs)
        loss_ingrs1 = tf.reduce_sum(loss_ingrs1)

        pooled_prbs = tf.gather(pooled_prbs, labels, batch_dims=1)
        pooled_prbs = tf.math.multiply(pooled_prbs, mask)  # ignore loss from eos and pad token
        loss_ingrs = bce(eos_head, pooled_prbs)
        # loss_ingrs = tf.math.multiply(loss_ingrs, mask)     # ignore loss from eos and pad token
        # loss_ingrs = tf.reduce_sum(loss_ingrs, axis=1)
        loss_ingrs = tf.reduce_mean(loss_ingrs)
        print(loss_ingrs)
        loss_ingrs = 0.7 * loss_ingrs + 0.3 * loss_ingrs1
        print(loss_ingrs)

        return loss_ingr_w * loss_ingrs + loss_eos_w * loss_eos


def train(model, train_image_paths, train_ings, train_ings_labels, mask, num_epochs=1, shuffle=True, teacher_forcing=False):
    for n in range(num_epochs):
        print("Epoch " + str(n))
        # shuffle inputs:
        indices = np.arange(train_ings.shape[0])
        if shuffle:
            np.random.shuffle(indices)
        for j in range(0, train_ings.shape[0] - model.batch_size, model.batch_size):
            batch_indices = indices[j:j + model.batch_size]
            batch_train_image_paths = [train_image_paths[idx] for idx in
                                       batch_indices]  # Have to do it like this because train_image is a list of strings now and can't use numpy's array indexing
            train_img = get_image_batch(batch_train_image_paths, is_train=True)
            train_ing = train_ings[batch_indices]
            labels = train_ings_labels[batch_indices]
            mask_lab = mask[batch_indices]
            with tf.GradientTape() as tape:
                # sampled_ids is None for Teacher Forcing
                sampled_ids, logits = model(train_img, train_ing, teacher_forcing=teacher_forcing)
                # USE THIS VERSION IF USING TEACHER FORCING
                # logits = model(train_img, train_ing, teacher_forcing=True)
                loss = model.loss(logits, labels, mask_lab, teacher_forcing=teacher_forcing)
                # USE THIS VERSION IF USING TEACHER FORCING:
                # loss = model.loss(logits, labels, mask_lab, teacher_forcing=True)
                print("loss at step" + str(j) + " = " + str(loss.numpy()))
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_image_paths, test_ings, test_ings_labels, num_epochs=1, shuffle=True):
    indices = np.arange(test_ings.shape[0])
    ings = []
    truth = []
    for j in range(0, test_ings.shape[0] - model.batch_size, model.batch_size):
        batch_indices = indices[j:j + model.batch_size]
        batch_test_image_paths = [test_image_paths[idx] for idx in
                                  batch_indices]  # Have to do it like this because train_image is a list of strings now and can't use numpy's array indexing
        test_img = get_image_batch(batch_test_image_paths, is_train=True)
        test_ing = test_ings[batch_indices]
        labels = test_ings_labels[batch_indices]
        start = [[START_INDEX]]
        ings.append(model(test_img, start)[0])
        truth.append(labels)
    ings = np.concatenate(ings, axis=0)
    truth = np.concatenate(truth, axis=0)
    print(np.array(ings).shape)
    print(np.array(truth).shape)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(ings.shape[0]):
        predings = ings[i]
        predings = predings[:int(np.where(predings == EOS_INDEX)[0])]
        trueings = truth[i]
        trueings = trueings[:int(np.where(trueings == EOS_INDEX)[0])]
        tpcur = 0
        fpcur = 0
        for ing in predings:
            if ing in trueings:
                tp += 1
                tpcur += 1
            else:
                fp += 1
                fpcur += 1
        fn += len(trueings) - tpcur
        # tn += len(vocab)-len(predings)
    f1 = float(tp) / (tp + (1.0 / 2.0 * (fp + fn)))
    print("F1 score is: " + str(f1))


def main():
    print("PREPROCESSING STARTING")
    train_image_paths, train_ingredients, test_image_paths, test_ingredients, vocab \
        = get_data(classes_path, ingredients_path, images_path, train_image_path, test_image_path)
    print("PREPROCESSING DONE")

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    print(train_ingredients.shape[0])

    model = Ingredient_Decoder(len(vocab), WINDOW_SZ)
    train_ings = train_ingredients[:, 0:20]
    train_ings_label = train_ingredients[:, 1:21]
    mask = np.ones([train_ingredients.shape[0], WINDOW_SZ], dtype=np.int64)
    mask[(train_ings_label == PAD_INDEX) ^ (train_ings_label == EOS_INDEX)] = 0
    train(model, train_image_paths, train_ings, train_ings_label, mask, num_epochs=1, teacher_forcing=True)
    # model is trained
    im_0 = get_image_batch([train_image_paths[0]], is_train=False)
    im_1 = get_image_batch([train_image_paths[127]], is_train=False)
    im_2 = get_image_batch([train_image_paths[435]], is_train=False)
    im_3 = get_image_batch([train_image_paths[302]], is_train=False)
    print(f"im_0 shape: {im_0.shape}")  # Should be (1, 224, 224, 3)
    start = [[START_INDEX]]
    ing_id_list_0 = model(im_0, start, sample_k=10)[0][0]  # len 20 list of ingredient IDs
    ing_id_list_1 = model(im_1, start, sample_k=10)[0][0]  # len 20 list of ingredient IDs
    ing_id_list_2 = model(im_2, start, sample_k=10)[0][0]  # len 20 list of ingredient IDs
    ing_id_list_3 = model(im_3, start, sample_k=10)[0][0]  # len 20 list of ingredient IDs

    def ingredient_ids_to_strings(id_list, from_tensor=False):
        if from_tensor:
            return [reverse_vocab[ing_id.numpy()] for ing_id in id_list]
        else:
            return [reverse_vocab[ing_id] for ing_id in id_list]

    print("Image 0 prediction:")
    print(ingredient_ids_to_strings(ing_id_list_0, from_tensor=True))
    print("Image 0 ground truth:")
    print(ingredient_ids_to_strings(train_ings_label[0]))
    plt.imshow(im_0[0])
    plt.show()

    print("Image 1 prediction:")
    print(ingredient_ids_to_strings(ing_id_list_1, from_tensor=True))
    print("Image 1 ground truth:")
    print(ingredient_ids_to_strings(train_ings_label[127]))
    plt.imshow(im_1[0])
    plt.show()

    print("Image 2 prediction:")
    print(ingredient_ids_to_strings(ing_id_list_2, from_tensor=True))
    print("Image 2 ground truth:")
    print(ingredient_ids_to_strings(train_ings_label[435]))
    plt.imshow(im_2[0])
    plt.show()

    print("Image 3 prediction:")
    print(ingredient_ids_to_strings(ing_id_list_3, from_tensor=True))
    print("Image 3 ground truth:")
    print(ingredient_ids_to_strings(train_ings_label[302]))
    plt.imshow(im_3[0])
    plt.show()

    # test_ings = test_ingredients[:,0:20]
    # test_ings_label = test_ingredients[:,1:21]
    # test(model, test_image_paths,test_ings,test_ings_label)


if __name__ == "__main__":
    main()
