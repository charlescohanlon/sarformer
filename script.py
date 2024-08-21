from PIL import Image
import os

ROOT = "/scratch/bdej/cohanlon/unsupervised/train/depth"
for dir in os.listdir(ROOT):
    for file in os.listdir(os.path.join(ROOT, dir)):
        path = os.path.join(ROOT, dir, file)
        im = Image.open(path)
        if im.size != (224, 224):
            print("Resizing", path)
            im = im.resize((224, 224), resample=Image.LANCZOS)
            im.save(path, subsampling=0, quality=95)

