import torch
from mocban_pix2pix.utils import save_checkpoint, load_checkpoint, test
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import torch.optim as optim
from mocban_pix2pix.base_config import base_config as config
import albumentations as A
from albumentations.pytorch import ToTensorV2
from mocban_pix2pix.generator_model import Generator
from mocban_pix2pix.discriminator_model import Discriminator
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt	
import numpy as np
from PIL import Image

torch.backends.cudnn.benchmark = True

disc = Discriminator(in_channels=config['CHANNELS_IMG']).to(config['DEVICE'])
gen = Generator(in_channels=config['CHANNELS_IMG'], features=64).to(config['DEVICE'])
opt_disc = optim.Adam(disc.parameters(), lr=config['LEARNING_RATE'], betas=(0.5, 0.999),)
opt_gen = optim.Adam(gen.parameters(), lr=config['LEARNING_RATE'], betas=(0.5, 0.999))


gen_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt_gen, patience=20, factor=0.5)

eval_augmentation = A.Compose(
[A.Resize(width=512, height=512),
 A.Normalize(mean=0.5, std=0.5, max_pixel_value=255.0, ),
 ToTensorV2(),
 ],
)

if config['LOAD_MODEL']:
	load_checkpoint(
	    config['CHECKPOINT_GEN'], gen, opt_gen, gen_scheduler,config['LEARNING_RATE'],
	)
gen.eval()

#TEST voi 1 anh duy nhat
#img_path = "/home/hmi/linhht/mocban_tool_full_version/hannom_anntotation_tool/static/characters/1.png"
#img_path2 = "/home/hmi/linhht/mocban_tool_full_version/hannom_anntotation_tool/static/characters/2.png"
#image = np.expand_dims(np.array(Image.open(img_path).convert('L')), axis=2)
#aug_image = torch.unsqueeze(eval_augmentation(image=image)["image"], dim=0).to("cuda")
#y_fake = gen(aug_image)* 0.5 + 0.5
#y_fake = torch.permute(make_grid(y_fake, nrow=1),(1,2,0)).cpu().detach().numpy() * 255
#cv2.imwrite(img_path2, y_fake)
print("-------------gen-----------------")


