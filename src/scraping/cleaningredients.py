import re
from pattern.en import singularize
import inflection as inf
blacklistf = open("blacklist.txt", "r")
blacklist = [l.rstrip() for l in blacklistf]
baseingredientsf = open("baseingredients.txt", "r")
baseingredients = [l.rstrip() for l in baseingredientsf][0].split(",")
def clean_ingredients(f):
    regex = re.compile("[\\d¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞]|\\(.*\\)|,.*|\\sfor\\s.*|\\s-.*")
    openedf = open(f, "r")
    ingredients = []
    for line in openedf:
        words = line.rstrip()
        words = re.sub(regex, "", words)
        words = words.split(" ")
        words = [inf.singularize(word) for word in words if word not in blacklist and word]
        str = ""
        for word in words:
            str += word + " "
        str = str[:-1]
        if str:
            ingredients.append(str.lower())
    return ingredients
outingredients = open("ingredients.txt", "w")
ingredientsarr = []
allingredients = []
for i in range(10718):
    out = []
    for ingredient in clean_ingredients("ingredients/" + str(i) + ".txt"):
        #if any([base in ingredient for base in baseingredients]):
        #outingredients.write(ingredient + ",")
        allingredients.append(ingredient)
        out.append(ingredient)
    ingredientsarr.append(out)
    #outingredients.write("\n")
vocab = {}
for ingredient in allingredients:
    if ingredient not in vocab:
        vocab[ingredient] = ingredient
print(len(vocab))
for ingredient in vocab:
    for ingredientmatch in vocab:
        if ingredient != ingredientmatch:
            ingredient_words = ingredient.split(" ")
            ingredientmatch_words = ingredientmatch.split(" ")
            if len(ingredient_words) >= 2 and len(ingredientmatch_words) >= 2:
                if ingredient_words[-2:]==ingredientmatch_words[-2:] or ingredient_words[:2]==ingredientmatch_words[:2]:
                    if len(ingredientmatch_words) < len(ingredient_words):
                        vocab[ingredient] = ingredientmatch
newvocab = {}
for word in vocab:
    if vocab[word] not in newvocab:
        newvocab[vocab[word]] = []
print(len(newvocab))
for word in newvocab:
    for word2 in newvocab:
        if word != word2:
            words1 = word.split(" ")
            words2 = word2.split(" ")
            if words1[0] == words2[0] or words1[0] == words2[-1]:
                newvocab[word].append(words1[0])
            elif words1[-1] == words2[0] or words1[-1] == words2[-1]:
                newvocab[word].append(words1[-1])
for word in newvocab:
    if newvocab[word]:
        newvocab[word] = max(set(newvocab[word]), key = newvocab[word].count)
    else:
        newvocab[word] = word
finalvocab = {}
for word in newvocab:
    if newvocab[word] not in finalvocab:
        finalvocab[newvocab[word]] = 0
print(len(finalvocab))
#for word in finalvocab:
#    outingredients.write(word)
#    outingredients.write("\n")
for ingredient in allingredients:
    finalvocab[newvocab[vocab[ingredient]]] += 1
for recipe in ingredientsarr:
    str = ""
    for ingredient in recipe:
        str += newvocab[vocab[ingredient]] + ","
    str = str[:-1]
    outingredients.write(str)
    outingredients.write("\n")
