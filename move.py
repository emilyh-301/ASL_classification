# https://pillow.readthedocs.io/en/stable/reference/Image.html
# When translating a color image to greyscale (mode “L”), the library uses the ITU-R 601-2 luma transform

import os
import sys
from PIL import Image

if len(sys.argv) != 4:
    print('\nSet the source path, destination path, and modulus correctly!\n')
    sys.exit(1)

src = sys.argv[1]
dest = sys.argv[2]
modulus = int(sys.argv[3])

img_size = (200, 200)
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

for category in categories:
    print(category)
    cat_src = os.path.join(src, category)
    if not os.path.isdir(cat_src):
        cat_src = os.path.join(src, category.lower())
    cat_dest = os.path.join(dest, category)
    if not os.path.exists(cat_dest):
        os.mkdir(cat_dest)

    counter = 1
    for filename in os.listdir(cat_src):
        src_img_path = os.path.join(cat_src, filename)
        if counter % modulus == 0:
            dest_img_path = os.path.join(cat_dest, filename)
            img = Image.open(src_img_path).resize(img_size).convert('L')
            img.save(dest_img_path)
        counter += 1
