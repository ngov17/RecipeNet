from PIL import Image
import numpy as np
import os
def resize_images(images):
    out = []
    for image in images:
        image = Image.fromarray(image)
        x, y = image.size
        if x > y:
            image = image.resize(((int(float(x)/float(y)*256)), 256))
        else:
            image = image.resize((256, (int(float(y)/float(x)*256))))
        x, y = image.size
        image = image.crop((int(float(x-224)/2.0), int(float(y-224)/2.0), 224+int(float(x-224)/2.0), 224+int(float(y-224)/2.0)))
        out.append(np.asarray(image))
