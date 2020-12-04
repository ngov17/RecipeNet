import requests
from bs4 import BeautifulSoup
maxpage = 400
alllinks = []
for i in range(maxpage):
    print(i)
    URL = "https://www.allrecipes.com/recipes/80/main-dish/?page=" + str(i)
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    results = soup.find_all("a",class_="tout__titleLink")
    for link in results:
        if link["href"].startswith("/recipe"):
            alllinks.append(link["href"])
print(len(alllinks))
with open('main-dish.txt', 'w') as filehandle:
    for listitem in alllinks:
        filehandle.write('%s\n' % listitem)
