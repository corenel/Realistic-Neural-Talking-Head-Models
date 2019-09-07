import torch
import cv2
from matplotlib import pyplot as plt

from loss.loss_discriminator import *
from loss.loss_generator import *
from network.blocks import *
from network.model import *
from image_demo.image_extraction_conversion import *
"""Init"""

#Paths
# path_to_model_weights = 'model_weights.tar'
num_ft = 40
path_to_model_weights = f'finetuned_model_{num_ft}.tar'
path_to_embedding = 'e_hat_images.tar'

device = torch.device("cuda:0")
cpu = torch.device("cpu")

checkpoint = torch.load(path_to_model_weights, map_location=cpu)
e_hat = torch.load(path_to_embedding, map_location=cpu)
e_hat = e_hat['e_hat'].to(device)

G = Generator(256, finetuning=True, e_finetuning=e_hat)
G.eval()
"""Training Init"""
G.load_state_dict(checkpoint['G_state_dict'])
G.to(device)
G.finetuning_init()
"""Main"""
image_id = '01'
image_path = f'inter_id/{image_id}_gt.jpg'

with torch.no_grad():
    x, g_y = generate_landmarks(image_path=image_path, device=device, pad=50)

    g_y = g_y.unsqueeze(0)
    x = x.unsqueeze(0)

    #forward
    # Calculate average encoding vector for video
    #f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2], f_lm.shape[-1]) #BxK,2,3,224,224
    #train G

    x_hat = G(g_y, e_hat)

    out1 = x_hat.transpose(1, 3)[0]
    out1 = out1.to(cpu).numpy()

    out2 = x.transpose(1, 3)[0]
    out2 = out2.to(cpu).numpy()

    out3 = g_y.transpose(1, 3)[0]
    out3 = out3.to(cpu).numpy()

    cv2.imwrite(f'{image_id}_samsung_t1_ft{num_ft}.jpg', cv2.cvtColor(out1, cv2.COLOR_BGR2RGB))
    # cv2.imwrite('me.jpg', cv2.cvtColor(out2, cv2.COLOR_BGR2RGB))
    # cv2.imwrite('ladnmark.jpg', cv2.cvtColor(out3, cv2.COLOR_BGR2RGB))
