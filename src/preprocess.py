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
trainimages = []
trainingredientlist = []
testimages = []
testingredientlist = []
trainlist = []
testlist = []
trainfile = open("annotations/train_images.txt", "r")
testfile = open("annotations/test_images.txt", "r")
for line in trainfile:
    trainlist.append("images/"+line.rstrip())
for line in testfile:
    testlist.append("images/"+line.rstrip())
for r,d,f in os.walk("images"):
    for file in f:
        name = file.split("_")[1:-1]
        str = ""
        for word in name:
            str += word + " "
        str = str[:-1]
        if str in ingredientsdict:
            if os.path.join(r, file) in trainlist:
                trainimages.append(np.asarray(Image.open(os.path.join(r, file))))
                trainingredientlist.append(ingredientsdict[str].split(","))
            elif os.path.join(r, file) in testlist:
                testimages.append(np.asarray(Image.open(os.path.join(r, file))))
                testingredientlist.append(ingredientsdict[str].split(","))
print(len(trainimages))
print(len(testimages))
