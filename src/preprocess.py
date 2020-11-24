import sys,os
import numpy as np
from PIL import Image
classfile = open("annotations/classes_Recipes5k.txt", "r")
classes = []
for line in classfile:
    classes.append(line.rstrip().lower())
ingredientsfile = open("annotations/ingredients_simplified_Recipes5k.txt", "r")
ingredients = []
for line in ingredientsfile:
    ingredients.append(line.rstrip().lower())
ingredientsdict = {}
for i in range(len(ingredients)):
    ingredientsdict[classes[i]] = ingredients[i]
images = []
ingredientlist = []
for r,d,f in os.walk("images"):
    for file in f:
        name = file.split("_")[1:-1]
        str = ""
        for word in name:
            str += word + " "
        str = str[:-1]
        if str in ingredientsdict:
            images.append(np.asarray(Image.open(os.path.join(r, file))))
            ingredientlist.append(ingredientsdict[str].split(","))
vocabulary = {}
inc = 0
for inglist in ingredientslist:
    for ingredient in ingredientslist:
        if not ingredient in vocabulary:
            vocabulary[ingredient] = inc
            inc += 1
