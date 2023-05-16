from .swin_transformer import SwinTransformer
from .vae import ImageVAE


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(img_size=(config.DATA.IMG_H,config.DATA.IMG_W),
                                patch_size=(config.MODEL.SWIN.PATCH_H,config.MODEL.SWIN.PATCH_W),
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                feature_size=config.MODEL.FEATURE_SIZE,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                avg_pool=config.MODEL.SWIN.AVG_POOL)
    elif model_type == 'vae':
        model = ImageVAE(input_channel=config.MODEL.VAE.IN_CHANS,
                feature_size=config.MODEL.VAE.FEATURE_SIZE,
                img_size=(config.DATA.IMG_H,config.DATA.IMG_W)
                )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
