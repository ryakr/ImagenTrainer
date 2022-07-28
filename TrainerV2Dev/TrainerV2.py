import wandb
from collections import deque

from imagen_pytorch import Unet, Imagen, ImagenTrainer, ElucidatedImagen
from imagen_pytorch.data import get_images_dataloader
from data_generator import ImageLabelDataset
from gan_utils import get_images, get_vocab
from torchvision import transforms
from torchvision.transforms import functional as VTF
from torchvision.utils import make_grid, save_image
import math
import numbers
import os
import time
import numpy as np
import torch

wandb.init(project="imagen-Goo")
wandb.config.scales = [64,256,1024]
wandb.config.batch_size = 64
wandb.config.max_batch_size = 16
wandb.config.unet_to_train = 2
wandb.config.shuffle = True
wandb.config.drop_tags=0.75
wandb.config.image_size = wandb.config.scales[wandb.config.unet_to_train-1]


# unets for unconditional imagen
source = '/workspace/GOO'
network = 'GOO_P1.pth'
imgs = get_images(source, verify=False)
txts = get_images(source, exts=".txt")

def txt_xforms(txt):
    # print(f"txt: {txt}")
    txt = txt.split(", ")
    if False:
        np.random.shuffle(txt)

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
    #random_crop_sizes = (None, 128, 256),
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
)
print('Trainer')
trainer = ImagenTrainer(imagen, fp16=True, dl_tuple_output_keywords_names=('images', 'texts'), split_valid_from_train=True).cuda()

tforms = transforms.Compose([
        PadImage(),
        transforms.Resize((wandb.config.image_size, wandb.config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
print('Data')

data = ImageLabelDataset(imgs, txts, None,
                        poses=None,
                        dim=(wandb.config.image_size, wandb.config.image_size),
                        transform=tforms,
                        tag_transform=txt_xforms,
                        channels_first=True,
                        return_raw_txt=True,
                        no_preload=False)
dl = torch.utils.data.DataLoader(data,
                                batch_size=wandb.config.batch_size,
                                shuffle=True,
                                num_workers=14,
                                pin_memory=True)

trainer.add_train_dataloader(dl)
print('Network')
trainer.load(network, noop_if_not_exist = True)

sample_texts=['canine, underwear',
              'felid, lion, animal genitalia ',
              'bear',
              'otter']


rate = deque([1], maxlen=5)
# working training loop
print('Scale: {} | Unet: {}'.format(wandb.config.image_size, wandb.config.unet_to_train))
for i in range(200000):
    t1 = time.monotonic()
    loss = trainer.train_step(unet_number = wandb.config.unet_to_train, max_batch_size = wandb.config.max_batch_size)
    t2 = time.monotonic()
    rate.append(round(1.0 / (t2 - t1), 2))
    wandb.log({"loss": loss, "step": i})
    print(f'loss: {loss} | Step: {i} | Rate: {round(np.mean(rate), 2)}')

    if not (i % 50) and False:
        valid_loss = trainer.valid_step(unet_number = wandb.config.unet_to_train, max_batch_size = wandb.config.max_batch_size)
        wandb.log({"Validated loss": valid_loss, "step": i})
        print(f'valid loss: {valid_loss}')

    if not (i % 250) and trainer.is_main: # is_main makes sure this can run in distributed
        rng_state = torch.get_rng_state()
        torch.manual_seed(1)
        images = trainer.sample(batch_size = 4, return_pil_images = False, texts=sample_texts, stop_at_unet_number=wandb.config.unet_to_train, return_all_unet_outputs=True) # returns List[Image]
        torch.set_rng_state(rng_state)
        sample_images0 = transforms.Resize(wandb.config.image_size)(images[0])
        sample_images1 = transforms.Resize(wandb.config.image_size)(images[-1])
        sample_images = torch.cat([sample_images0, sample_images1])
        grid = make_grid(sample_images, nrow=4, normalize=False, range=(-1, 1))
        VTF.to_pil_image(grid).save(f'./sample-{i // 100}.png')
        print('SAVING NETWORK')
        trainer.save(network)
        print('Network Saved')
        images = wandb.Image(VTF.to_pil_image(grid), caption="Top: Unet1, Bottom: Unet{}".format(wandb.config.unet_to_train))
        wandb.log({ "step": i, "outputs": images})