from PIL import Image
import numpy as np
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
            x, y = image.size
            image = image.crop((int(float(x-224)/2.0), int(float(y-224)/2.0), x-int(float(x-224)/2.0), y-int(float(y-224)/2.0)))
            image.save(os.path.join(r, file))
