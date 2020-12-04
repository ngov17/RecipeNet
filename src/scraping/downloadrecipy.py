from recipe_scrapers import scrape_me
import requests
filen = "main-dish.txt"
openedfile = open(filen, "r")
recipefile = "recipes.txt"
ingredientsfile = "ingredients.txt"
links = [link for link in openedfile]
for i in range(8492,len(links)):
    link = links[i]
    print(i)
    name = 'https://www.allrecipes.com' + link.rstrip()
    scraper = scrape_me(name)
    if scraper.image() != 'https://imagesvc.meredithcorp.io/v3/mm/image?url=https%3A%2F%2Fwww.allrecipes.com%2Fimg%2Fmisc%2Fog-default.png':
        img_data = requests.get(scraper.image()).content
        with open("images/" + str(i) + '.jpg', 'wb') as handler:
            handler.write(img_data)
        with open("recipes/" + str(i) + '.txt', "w") as openedrecipe:
            openedrecipe.write(scraper.instructions())
        with open("ingredients/" + str(i) + '.txt', 'w') as filehandle:
            for listitem in scraper.ingredients():
                filehandle.write('%s\n' % listitem)
