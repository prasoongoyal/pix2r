import torch
import numpy as np
from PIL import Image
import glob

FRAME_DIR = 'metaworld-dataset/'
OUTPUT_DIR = 'metaworld-dataset/processed-frames'
N_OBJ = 13
N_ENV = 100

def main(camera):
  for obj in range(13):
    for env in range(100):
      img_files = glob.glob('{}/obj{}-env{}/{}*.png'.format(FRAME_DIR, obj, env, camera))
      result = []
      for f in img_files:
        img = Image.open(f)
        img = img.resize((50, 50))
        img = np.array(img)
        result.append(img)
      result = np.transpose(result, (0, 3, 1, 2)) / 255.
      torch.save(result, open('{}/obj{}-env{}-{}-50x50.pt'.format(
          OUTPUT_DIR, obj, env, camera), 'wb'))


if __name__ == '__main__':
  import sys
  main(sys.argv[1])

