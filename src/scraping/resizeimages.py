from PIL import Image
for i in range(2776):
    image = Image.open("images/" + str(i) + ".jpg").convert('RGB')
    x, y = image.size
    if x > y:
        image = image.resize(((int(float(x)/float(y)*256)), 256))
    else:
        image = image.resize((256, (int(float(y)/float(x)*256))))
    #image = image.crop((int(float(x-224)/2.0), int(float(y-224)/2.0), x-int(float(x-224)/2.0), y-int(float(y-224)/2.0)))
    image.save("images/" + str(i) + ".jpg")
