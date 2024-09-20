import lavis.models.utils.paramUtil as paramUtil
from lavis.models.utils.plot_script import *

from lavis.models.networks.modules import *
from lavis.models.networks.quantizer import *

def loadVQModel():
    vq_encoder = VQEncoderV3(263 - 4, [1024,1024], 2)
    # vq_decoder = VQDecoderV3(opt.dim_vq_latent, dec_channels, opt.n_resblk, opt.n_down)
    quantizer = Quantizer(1024, 1024, 1)
    checkpoint = torch.load('../MA_LMM/checkpoint/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/model/finest.tar',map_location='cuda')
    vq_encoder.load_state_dict(checkpoint['vq_encoder'])
    quantizer.load_state_dict(checkpoint['quantizer'])
    return vq_encoder, quantizer