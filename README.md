# Scripts for running imagen-pytorch. Based on DeepGlugs trainer.

This is currently in development, so things will probably break.

## Setup:
```bash
python3 -m pip install imagen-pytorch
```

## Training:

Currently, this is set up to use danbooru-style tags such as:

```
1girl, blue_dress, super_artist
```

The dataloader expects a directory with images and tags laid out like this:

```
dataset/
   tags/img1.txt
   tags/img2.txt
   ...
   imgs/img1.png
   imgs/img2.png
```

The subdirectories doesn't really matter, only the filenames matter.

### To train:

```bash
python3 imagen-3net.py --train --source /path/to/dataset --imagen yourmodel.pth
```

Changable settings are in the script, such as unet number and the image size to resize to.