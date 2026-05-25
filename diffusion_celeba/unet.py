from diffusers import UNet2DModel


def build_model(model_size):
    in_channels = 3
    height = 64
    
    if model_size == 'small':
        # 63M model
        channels = (64, 128, 256, 512)
        up_blocks = ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
        down_blocks = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")

    elif model_size == 'large':
        # 109M model
        channels = (128, 128, 256, 512, 512)
        up_blocks = ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
        down_blocks = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")

    else:
        raise ValueError("model_size must be either 'small' or 'large'")

    model = UNet2DModel(
        sample_size=height,
        in_channels=in_channels,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=channels,
        down_block_types=down_blocks,
        up_block_types=up_blocks
        )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    return model
