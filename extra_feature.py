import torch

from lavis.models.eva_vit import *
from lavis.models.clip_vit import *
import os
import h5py
import numpy as np
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import shutil
from lavis.processors.blip_processors import Blip2VideoTrainProcessor
import contextlib
from transformers import CLIPFeatureExtractor, CLIPVisionModel
def maybe_autocast(dtype=torch.float16):
    # if on cpu, don't use autocast
    # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
    enable_autocast = 'cuda' != torch.device("cpu")

    if enable_autocast:
        return torch.cuda.amp.autocast(dtype=dtype)
    else:
        return contextlib.nullcontext()
def init_vision_encoder(model_name, img_size, drop_path_rate, use_grad_checkpoint, precision):
    assert model_name in [
        "eva_clip_g",
        "eva2_clip_L",
        "clip_L",
    ], "vit model must be eva_clip_g, eva2_clip_L or clip_L"
    if model_name == "eva_clip_g":
        visual_encoder = create_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision
        )
    # elif model_name == "eva2_clip_L":
    #     visual_encoder = create_eva2_vit_L(
    #         img_size, drop_path_rate, use_grad_checkpoint, precision
    #     )
    elif model_name == "clip_L":
        visual_encoder = create_clip_vit_L(img_size, use_grad_checkpoint, precision)
    ln_vision = LayerNorm(visual_encoder.num_features)
    vit_name = model_name
    return visual_encoder, ln_vision

ts=Blip2VideoTrainProcessor(224)
visual_encoder, ln_vision = init_vision_encoder("eva_clip_g", 224, 0, False, 'fp32')

#encoder_name = 'models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
#encoder_name = 'clip-vit-large-patch14'
# feature_extractor = CLIPFeatureExtractor.from_pretrained(encoder_name)
# visual_encoder = CLIPVisionModel.from_pretrained(encoder_name).to('cuda')

for name, param in visual_encoder.named_parameters():
    param.requires_grad = False
visual_encoder = visual_encoder.eval()
visual_encoder.to('cuda')
ln_vision.to('cuda')


names=os.listdir('data/msvd/frames')
inde=0
for name in names:
    pictures=os.listdir('data/msvd/frames/'+name)
    pictures.sort()
    frame_list=[]
    for img in pictures:
        frame = Image.open('data/msvd/frames/'+name+'/'+img).convert("RGB")
        frame = pil_to_tensor(frame)
        frame_list.append(frame)
    video = torch.stack(frame_list, dim=1)
    video = video.float()
    video = ts(video)
    video=video.permute(1,0,2,3)
    video=video.half().to('cuda')

    # with torch.no_grad():
    #     image_embeds = visual_encoder(pixel_values=video).last_hidden_state
    with maybe_autocast():
        image_embeds = ln_vision(visual_encoder(video))

    #image_embeds=image_embeds.to(torch.float32)

    image_embeds=image_embeds.cpu()
    try:
        os.mkdir('data/msvd/tensor'+'/'+name)
    except:
        pass
    for i in range(len(image_embeds)):
        # print("frame{:06d}.pt".format(i + 1))
        # print(image_embeds[i].shape)
        # print('data/msvd/tensor/'+name+'/'+"frame{:06d}.pt".format(i + 1))
        image_emb=image_embeds[i].detach().numpy()
        np.save('data/msvd/tensor/'+name+'/'+"frame{:06d}.npy".format(i + 1),image_emb)
        #torch.save(image_embeds[i],'data/msvd/tensor/'+name+'/'+"frame{:06d}.pt".format(i + 1))
    print(inde)
    inde+=1
    # print(image_embeds.shape,name)
    # torch.save(image_embeds,'data/msvd/tensor/'+name+'.pt')
    
# print(visual_encoder)
# image_embeds = ln_vision(visual_encoder(image))