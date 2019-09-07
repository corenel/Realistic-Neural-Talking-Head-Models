"""Main"""
import torch

from dataset.video_extraction_conversion import select_frames, select_image_frame, generate_cropped_landmarks
from network.blocks import *
from network.model import Embedder

import numpy as np
"""Hyperparameters and config"""
device = torch.device("cuda:0")
cpu = torch.device("cpu")
path_to_e_hat_images = 'e_hat_images.tar'
path_to_chkpt = 'model_weights.tar'
image_id = '01'
path_to_image = f'inter_id/{image_id}_src.jpg'
T = 32
"""Loading Embedder input"""
frame_mark_images = select_image_frame(path_to_image)
frame_mark_images = generate_cropped_landmarks(frame_mark_images, pad=50)
frame_mark_images = torch.from_numpy(np.array(frame_mark_images)).type(
    dtype=torch.float)  #T,2,256,256,3
frame_mark_images = frame_mark_images.transpose(2, 4).to(device)  #T,2,3,256,256
f_lm_images = frame_mark_images.unsqueeze(0)  #1,T,2,3,256,256

E = Embedder(256).to(device)
E.eval()
"""Loading from past checkpoint"""
checkpoint = torch.load(path_to_chkpt, map_location=cpu)
E.load_state_dict(checkpoint['E_state_dict'])
"""Inference"""
with torch.no_grad():
    #forward
    # Calculate average encoding vector for image
    f_lm = f_lm_images
    f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2],
                             f_lm.shape[-1])  #BxT,2,3,224,224
    e_vectors = E(f_lm_compact[:, 0, :, :, :],
                  f_lm_compact[:, 1, :, :, :])  #BxT,512,1
    e_vectors = e_vectors.view(-1, f_lm.shape[1], 512, 1)  #B,T,512,1
    e_hat_images = e_vectors.mean(dim=1)

print('Saving e_hat...')
torch.save({'e_hat': e_hat_images}, path_to_e_hat_images)
print('...Done saving')
