import sys, os
import numpy as np
from PIL import Image
from skimage import io, transform, img_as_float32
import random

START_TOKEN = "*START*" #Index: 3
STOP_TOKEN = "*STOP*"   #Index: 0
PAD_TOKEN = "*PAD*"     #Index: 1
UNK_TOKEN = "*UNK*"     #Index: 2
WINDOW_SIZE = 20

def preprocess_image(image_path, is_train=True):
    """
    1. read image
    2. resize image to 256 * 256 * 3
    3. randomly sample a 224 * 224 *3 patch of the image
    4. normalize image intensity? (NOTE: not currently doing this, which is fine since original paper doesn't)
    5. return processed image
    """
    image = io.imread(image_path)
    h,w,c = image.shape
    newshape = (256,256,3)
    if h > w:
        newshape = (int((256.0 / float(w)) * float(h)), 256, c)
    elif h < w:
        newshape = (256, int((256.0 / float(h)) * float(w)), c)
    image = transform.resize(image, newshape, anti_aliasing=True)
    start_r = 0
    start_c = 0
    if is_train:
        start_r = random.randint(0, newshape[0]-224)
        start_c = random.randint(0, newshape[1]-224)
    else:
        start_r = (newshape[0] - 224) / 2
        start_c = (newshape[1] - 224) / 2
    image = image[start_r:start_r+224, start_c:start_c+224, :]
    return image

def get_image_batch(image_paths, is_train=True):
    """
    param image_paths: a list of paths to image locations (such as a length batch_size slice of a larger list)
    param is_train: True if processing images for training (sample random image patch) or False if processing for testing (sample central image patch)
    return: a numpy array of size (len(image_paths), 224, 224, 3) containing the preprocessed images
    """
    return np.stack([preprocess_image(path, is_train) for path in image_paths], axis=0)

def pad_ingredients(ingredient_list):
    """
    """
    return [[START_TOKEN] + line[:(WINDOW_SIZE - 2)] + [STOP_TOKEN] + [PAD_TOKEN] * (WINDOW_SIZE - len(line[:(WINDOW_SIZE - 2)]) - 1) for line in ingredient_list]

def convert_to_id(vocab, sentences):
    """
    """
    return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])

def build_vocab(ingredients):
    """
    """
    vocab = {STOP_TOKEN: 0, PAD_TOKEN: 1, UNK_TOKEN: 2, START_TOKEN: 3}
    tokens = []
    for i in ingredients: tokens.extend(i)
    all_words = sorted(list(set(tokens)))
    for i, word in enumerate(all_words):
        vocab[word] = i + 4
    return vocab

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

    train_ingredient_list = []
    test_ingredient_list = []

    train_image_paths = []
    test_image_paths = []

    train_file = open(train_image_path, "r")
    test_file = open(test_image_path, "r")

    for line in train_file:
        splitline = line.rstrip().split('/')
        train_image_paths.append(os.path.join(images, splitline[0], splitline[1]))
    for line in test_file:
        splitline = line.rstrip().split('/')
        test_image_paths.append(os.path.join(images, splitline[0], splitline[1]))

    for r, d, f in os.walk(images):
        for file in f:
            name = file.split("_")[1:-1]
            str = ""
            for word in name:
                str += word + " "
            str = str[:-1]
            if str in ingredients_dict:
                pth = os.path.join(r, file)
                if pth in train_image_paths:
                    train_ingredient_list.append(ingredients_dict[str].split(","))
                elif pth in test_image_paths:
                    test_ingredient_list.append(ingredients_dict[str].split(","))

    vocab = build_vocab(train_ingredient_list + test_ingredient_list)
    padded_train_ingredients = np.array(pad_ingredients(train_ingredient_list))
    padded_test_ingredients = np.array(pad_ingredients(test_ingredient_list))

    train_ingredients = convert_to_id(vocab, padded_train_ingredients)
    test_ingredients = convert_to_id(vocab, padded_test_ingredients)

    return train_image_paths, train_ingredients, test_image_paths, test_ingredients, vocab