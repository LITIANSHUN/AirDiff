from diffusers import DDPMScheduler, UNet1DModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from models.unet.unet_1d_condition import UNet1DConditionModel

def get_model(latent_dim=32):

    # return UNet1DModel(
    #         sample_size=latent_dim,
    #         in_channels=1,
    #         out_channels=1,
    #         layers_per_block=2,
    #         block_out_channels=(64, 128),
    #         down_block_types=("DownBlock1D", "DownBlock1D"),
    #         up_block_types=("UpBlock1D", "UpBlock1D"),
    #         # 类别

    #     )

    # if config['model_name'] == "UNet1D_label":
    return UNet1DConditionModel(
        sample_size=latent_dim,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 64, 128, 256),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock1D",
            "AttnDownBlock1D", 
            "CrossAttnDownBlock1D",
            "ResnetDownsampleBlock1D",
        ),
        mid_block_type="UNetMidBlock1DCrossAttn",
        up_block_types=(
            "ResnetUpsampleBlock1D",
            "CrossAttnUpBlock1D",
            "AttnUpBlock1D", 
            "UpBlock1D",
        ),
        # num_class_embeds=11,
        # class_embed_type = "projection",
        # projection_class_embeddings_input_dim = latent_dim,
        encoder_hid_dim=latent_dim
        # class_embeddings_concat=True, # whether to concatenate the class embeddings with the hidden states,
        # encoder_hid_dim = config["guidance_input_dim"], # the dimension of the guidance input 包括成功了没，分数。
        # # 目前的话就是到达目的地了没，拿到得分点没，拿到了多少分，那维度就是3，行开整
        # encoder_hid_dim_type="text_proj",

    )



