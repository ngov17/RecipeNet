from PIL import Image
import os
for r,d,f in os.walk("images"):
    for file in f:
        if file.endswith(".jpg"):
            image = Image.open(os.path.join(r, file)).convert("RGB")
            x, y = image.size
            if x > y:
                image = image.resize(((int(float(x)/float(y)*256)), 256))
            else:
                image = image.resize((256, (int(float(y)/float(x)*256))))
            image.save(os.path.join(r, file))
