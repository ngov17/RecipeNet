import sys, os
import numpy as np
from PIL import Image

data_path = os.path.join("..", "data")
classes_path = os.path.join(data_path, "classes_Recipes5k.txt")
ingredients_path = os.path.join(data_path, "ingredients_simplified_Recipes5k.txt")
images = os.path.join(data_path, "images")
train_image_path = os.path.join(data_path, "train_images.txt")
test_image_path = os.path.join(data_path, "test_images.txt")

PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
WINDOW_SIZE = 20


def resize_images(images):
    """
    """

    out = np.zeros((len(images), 224, 224, 3))
    i = 0
    for image in images:
        image = Image.fromarray(image)
        x, y = image.size
        if x > y:
            image = image.resize(((int(float(x)/float(y)*256)), 256))
        else:
            image = image.resize((256, (int(float(y)/float(x)*256))))
        x, y = image.size
        image = image.crop((int(float(x-224)/2.0), int(float(y-224)/2.0), 224+int(float(x-224)/2.0), 224+int(float(y-224)/2.0)))
        out[i] = np.asarray(image)
        i += 1
    return out


def normalize_images(images):
    """
    Normalizes each pixel in an image to a value between 0-1 by dividing each pixel by 255.0
    """

    return images / 255.0

def pad_ingredients(ingredient_list):
    """
    """

    padded_ingredients_list = []
    for line in ingredient_list:
        padded_ing = line[:(WINDOW_SIZE - 2)]
        padded_ing = [START_TOKEN] + padded_ing + [STOP_TOKEN] + [PAD_TOKEN] * (
                WINDOW_SIZE - len(padded_ing) - 1)
        padded_ingredients_list.append(padded_ing)

    return padded_ingredients_list

def convert_to_id(vocab, sentences):
    """
    """
    return np.stack(
        [[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])


def build_vocab(ingredients):
    """
    """

    tokens = []
    for i in ingredients: tokens.extend(i)
    all_words = sorted(list(set([STOP_TOKEN, PAD_TOKEN, UNK_TOKEN] + tokens)))

    vocab = {word: i for i, word in enumerate(all_words)}

    return vocab, vocab[PAD_TOKEN]


def get_data(classes_path, ingredients_path, images, train_image_path, test_image_path):
    """
    """

    class_file = open(classes_path, "r")
    classes = []
    for line in class_file:
        classes.append(line.rstrip().lower())
    ingredients_file = open(ingredients_path, "r")
    ingredients = []
    for line in ingredients_file:
        ingredients.append(line.rstrip().lower())
    ingredients_dict = {}
    for i in range(len(ingredients)):
        ingredients_dict[classes[i]] = ingredients[i]

    train_images = []
    train_ingredient_list = []

    test_images = []
    test_ingredient_list = []

    train_list = []
    test_list = []

    train_file = open(train_image_path, "r")
    test_file = open(test_image_path, "r")

    for line in train_file:
        splitline = line.rstrip().split('/')
        train_list.append(os.path.join(images, splitline[0], splitline[1]))
    for line in test_file:
        splitline = line.rstrip().split('/')
        test_list.append(os.path.join(images, splitline[0], splitline[1]))

    for r, d, f in os.walk(images):
        for file in f:
            name = file.split("_")[1:-1]
            str = ""
            for word in name:
                str += word + " "
            str = str[:-1]
            if str in ingredients_dict:
                pth = os.path.join(r, file)
                if pth in train_list:
                    train_images.append(np.asarray(Image.open(pth)))
                    train_ingredient_list.append(ingredients_dict[str].split(","))
                elif pth in test_list:
                    test_images.append(np.asarray(Image.open(pth)))
                    test_ingredient_list.append(ingredients_dict[str].split(","))

    # resize images to (224, 224, 3)
    train_images = normalize_images(resize_images(train_images))
    test_images = normalize_images(resize_images(test_images))

    vocab, pad_token_idx = build_vocab(train_ingredient_list + test_ingredient_list)
    padded_train_ingredients = np.array(pad_ingredients(train_ingredient_list))
    padded_test_ingredients = np.array(pad_ingredients(test_ingredient_list))

    train_ingredients = convert_to_id(vocab, padded_train_ingredients)
    test_ingredients = convert_to_id(vocab, padded_test_ingredients)

    return train_images, train_ingredients, test_images, test_ingredients, vocab, pad_token_idx


# test
# train_image, train_ingredients, test_image, test_ingredients, vocab, pad_token_idx \
#     = get_data(classes_path, ingredients_path, images, train_image_path, test_image_path)
# shapes and sizes of result:
# print(train_image.shape)
# print(train_ingredients.shape)
# print(test_image.shape)
# print(test_ingredients.shape)
# print(len(vocab))
# print(pad_token_idx)