import math
import numbers
import os
import random
import time
from collections import deque

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as VTF
from torchvision.utils import make_grid, save_image
from PIL import Image

from imagen_pytorch import Unet, Imagen, ImagenTrainer, ElucidatedImagen
from gan_utils import get_images, get_vocab
from data_generator import ImageLabelDataset


scales=[64,256,1024]
unetnum = 1
random_drop_tags=0.75

def txt_xforms(txt):
    # print(f"txt: {txt}")
    txt = txt.split(", ")
    if True:
        np.random.shuffle(txt)

    r = int(len(txt) * random_drop_tags)

    if r > 0:
        rand_range = random.randrange(r)

    if random_drop_tags > 0.0 and r > 0:
        txt.pop(rand_range)    
    
    txt = ", ".join(txt)

    return txt

def get_padding(image):    
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class PadImage(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return VTF.pad(img, get_padding(img), self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=None, help="image source")
    parser.add_argument('--tags_source', type=str, default=None, help="tag files. will use --source if not specified.")
    parser.add_argument('--poses', type=str, default=None)
    parser.add_argument('--tags', type=str, default=None)
    parser.add_argument('--vocab', default=None)
    parser.add_argument('--size', default=256, type=int)
    parser.add_argument('--num_unets', default=1, type=int, help="additional unet networks")
    parser.add_argument('--vocab_limit', default=None, type=int)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--imagen', default="imagen.pth")
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--replace', action='store_true', help="replace the output file")
    parser.add_argument('--unet_dims', default=128, type=int)
    parser.add_argument('--unet2_dims', default=128, type=int)

    # training
    parser.add_argument('--batch_size', default=348, type=int)
    parser.add_argument('--micro_batch_size', default=64, type=int)
    parser.add_argument('--samples_out', default="samples")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--shuffle_tags', action='store_true')
    parser.add_argument('--dp', action='store_true')
    parser.add_argument('--train_unet', type=int, default=0)
    parser.add_argument('--random_drop_tags', type=float, default=0.75)


    args = parser.parse_args()

    if args.tags_source is None:
        args.tags_source = args.source

    if args.vocab is None:
        args.vocab = args.source
    else:
        assert os.path.isfile(args.vocab)

    if args.train:
        train(args)
    else:
        sample(args)

def sample(args):

    if os.path.isfile(args.output) and not args.replace:
        return

    try:
        imagen = load(args, args.imagen)
    except:
        print(f"Error loading model: {args.imagen}")
        return

    cond_image = None

    if args.poses is not None and os.path.isfile(args.poses):
        tforms = transforms.Compose([PadImage(),
                                     transforms.Resize((args.size, args.size)),
                                     transforms.ToTensor()])
        cond_image = Image.open(args.poses)
        cond_image = tforms(cond_image).to(imagen.device)
        cond_image = cond_image.view(1, *cond_image.shape)

    sample_images = imagen.sample(texts=[args.tags],
                                  cond_images=cond_image,
                                  cond_scale=7.,
                                  return_pil_images=True)

    sample_images[0].save(args.output)

def restore_parts(state_dict_target, state_dict_from):
    for name, param in state_dict_from.items():
        if name not in state_dict_target:
            continue
        # if isinstance(param, Parameter):
        #    param = param.data
        if param.size() == state_dict_target[name].size():
            state_dict_target[name].copy_(param)
        else:
            print(f"layer {name}({param.size()} different than target: {state_dict_target[name].size()}")

    return state_dict_target

def save(imagen, path):
    out = {}
    unets = []
    for unet in imagen.unets:
        unets.append(unet.cpu().state_dict())
    out["unets"] = unets

    out["imagen"] = imagen.cpu().state_dict()

    torch.save(out, path)


def load(args, path):

    to_load = torch.load(path)

    unet1_dims = to_load.get("unet1_dims", args.unet_dims)
    unet2_dims = to_load.get("unet2_dims", args.unet2_dims)

    imagen = get_imagen(args, unet1_dims, unet2_dims)

    try:
        imagen.load_state_dict(to_load["model"], strict=True)
    except RuntimeError:
        print("Failed loading state dict. Trying partial load")
        imagen.load_state_dict(restore_parts(imagen.state_dict(),
                                             to_load))

    return imagen


def get_imagen(args, unet_dims=None, unet2_dims=None):

    if unet_dims is None:
        unet_dims = args.unet_dims

    if unet2_dims is None:
        unet2_dims = args.unet2_dims

    if args.poses is not None:
        cond_images_channels = 3
    else:
        cond_images_channels = 0

    unet1 = Unet(
    # unet for imagen
		cond_on_text = True,
		dim_mults = [1, 2, 3, 4],
        cond_dim=512,
		dim = 256,
		layer_attns = [False, True, True, True],
		layer_cross_attns = [False, True, True, True],
    )

    unets = [unet1]


    unet2 = Unet(
        cond_on_text = True,
        dim_mults = [1, 2, 4, 8],
        dim = 128,
        cond_dim=512,
        layer_attns = [False, False, False, True],
        layer_cross_attns = [False, False, False, True],
        memory_efficient = True,
    )
    
    unet3 = Unet(
        cond_on_text = True,
        dim_mults = [1, 2, 4, 8],
        dim = 128,
        cond_dim=512,
        layer_attns = [False, False, False, True],
        layer_cross_attns = [False, False, False, True],
        memory_efficient = True,
    )


    imagen = ElucidatedImagen(
        unets = (unet1, unet2, unet3),
        image_sizes = (64, 256, 1024),
        cond_drop_prob = 0.1,
        random_crop_sizes = (None, 128, 256),
        num_sample_steps = (64, 32, 32), # number of sample steps - 64 for base unet, 32 for upsampler (just an example, have no clue what the optimal values are)
        sigma_min = 0.002,           # min noise level
        sigma_max = (80, 160, 160),       # max noise level, @crowsonkb recommends double the max noise level for upsampler
        sigma_data = 0.5,            # standard deviation of data distribution
        rho = 7,                     # controls the sampling schedule
        P_mean = -1.2,               # mean of log-normal distribution from which noise is drawn for training
        P_std = 1.2,                 # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn = 80,                # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin = 0.05,
        S_tmax = 50,
        S_noise = 1.003,
        auto_normalize_img = True,
    ).cuda()
 

    return imagen


def train(args):

    scale=scales[(unetnum-1)]
    print('Scale: {} | Unet: {}'.format(scale, unetnum))
    imagen = get_imagen(args)

    trainer = ImagenTrainer(imagen, fp16=True)

    if args.dp:
        imagen = torch.nn.DataParallel(trainer)

    if args.imagen is not None and os.path.isfile(args.imagen):
        print(f"Loading model: {args.imagen}")
        imagen = load(args, args.imagen)
        trainer.load(args.imagen)
    
    print('Data')
    imgs = get_images(args.source, verify=False)
    txts = get_images(args.tags_source, exts=".txt")
    #vocab = get_vocab(args.vocab, top=args.vocab_limit)

    poses = None
    has_poses = False

    if args.poses is not None:
        poses = get_images(args.poses)
        has_poses = True

    tforms = transforms.Compose([
            PadImage(),
            transforms.Resize((scale, scale)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

    data = ImageLabelDataset(imgs, txts, None,
                             poses=poses,
                             dim=(scale,scale),
                             transform=tforms,
                             tag_transform=txt_xforms,
                             channels_first=True,
                             return_raw_txt=True,
                             no_preload=False)

    dl = torch.utils.data.DataLoader(data,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=14,
                                     pin_memory=True)

    disp_size = min(args.batch_size, 4)
    rate = deque([1], maxlen=5)

    os.makedirs(args.samples_out, exist_ok=True)

    sample_texts=['canine, underwear',
                  'felid, lion, animal genitalia ',
                  'bear',
                  'otter']


    sample_poses = None
    print('Epoch')
    
    for epoch in range(1, args.epochs + 1):
        step = 0
        for data in dl:

            if has_poses:
                images, texts, poses = data
            else:
                images, texts = data
                poses = None

            step += 1

            t1 = time.monotonic()
            losses = 0
            for i in range(1): #range(args.train_unet, len(imagen.unets))
                
                loss = trainer(
                    images,
                    cond_images=poses,
                    texts = texts,
                    unet_number = unetnum,
                    max_batch_size = args.micro_batch_size
                )

                trainer.update(unet_number=unetnum)

                losses = loss

            t2 = time.monotonic()
            rate.append(round(1.0 / (t2 - t1), 2))

            if step % 1 == 0:
                print("epoch {}/{} step {}/{} loss: {} - {}it/s - incriment {} Step Full {}".format(
                      epoch,
                      args.epochs,
                      step * args.batch_size,
                      len(imgs),
                      losses,
                      round(np.mean(rate), 2),
                      step/100, 
                      step))
                
            if epoch % 50 == 0 and step == 1:
                if poses is not None and sample_poses is None:
                    sample_poses = poses[:disp_size]

                sample_images = trainer.sample(texts=sample_texts,
                                               cond_images=sample_poses,
                                               cond_scale=3.,
                                               return_all_unet_outputs=True,
                                               stop_at_unet_number=unetnum)

                sample_images0 = transforms.Resize(scale)(sample_images[0])
                sample_images1 = transforms.Resize(scale)(sample_images[-1])
                sample_images = torch.cat([sample_images0, sample_images1])

                if poses is not None:
                    sample_poses0 = transforms.Resize(args.size)(sample_poses)
                    sample_images = torch.cat([sample_images.cpu(), sample_poses0.cpu()])

                grid = make_grid(sample_images, nrow=disp_size, normalize=False, range=(-1, 1))
                VTF.to_pil_image(grid).save(os.path.join(args.samples_out, f"imagen_{epoch}_{int(step / epoch)}.png"))

                if args.imagen is not None:
                    trainer.save(args.imagen, unet1_dims=args.unet_dims,
                                 unet2_dims=args.unet2_dims)


if __name__ == "__main__":
    main()