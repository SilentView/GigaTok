import os
import yaml
from torchvision.datasets.utils import download_url
import torch
import torchvision.models as torchvision_models
import timm

from tokenizer.tokenizer_image.vq.vq_vit_model import VQVitModelPlusArgs, VQVitModelPlus, VQVitModel2DPlusArgs, VQVitModel2DPlus

from tokenizer.tokenizer_image.vae.vae_vit_model import VAEVitModel, VAEVitModelArgs


def load_model_from_config(config):
    if isinstance(config, str):
        if not os.path.exists(config):
            raise ValueError(f"config file {config} does not exist.")
        else:
            with open(config, 'r') as file:
                config = yaml.safe_load(file)
    elif isinstance(config, dict):
        pass
    
    if config["trainer"]["distill_loss"]:
        emb_dim = config["model"]["init_args"].get("out_inner_dim", None)
        distill_model = config["trainer"].get("distill_model", "dinov2-vit-b")
        if emb_dim is None:
            emb_dim_map = {
                "dinov2-vit-l": 1024,
                "dinov2-vit-b": 768,
                "dinov2-vit-s": 384,
                "siglip-sovit-400m": 1152,
                "clip_dfn-vit-l/14": 1024,
            }
            emb_dim = emb_dim_map[distill_model]
            config["model"]["init_args"]["out_inner_dim"] = emb_dim
                
    # load the model
    model_cls = eval(config["model"]["model_cls"])
    if config["model"]["model_cls"] == "VQVitModelPlus":
        model_args = VQVitModelPlusArgs(**config["model"]["init_args"])
    elif config["model"]["model_cls"] == "VQVitModel2DPlus":
        model_args = VQVitModel2DPlusArgs(**config["model"]["init_args"])
    elif config["model"]["model_cls"] == "VAEVitModel":
        model_args = VAEVitModelArgs(**config["model"]["init_args"])
    else:
        raise NotImplementedError
    model = model_cls(model_args)
    print(model_args)

    return model


def download_load_multiprocess(repo, model):
    import torch.distributed as dist
    success_flag = False
    node_rank = int(os.environ.get('NODE_RANK', 0))

    try:
        rank = dist.get_rank()
    except:
        rank = 0

    if node_rank == 0 and rank == 0:
        model = torch.hub.load(repo, model)
        success_flag = True
    else:
        # wait until the master process downloads the model
        pass

    if torch.distributed.is_initialized():
        dist.barrier()

    if not success_flag:
        # for the non master process, load the model later
        model = torch.hub.load(repo, model)
        success_flag = True
    
    return model


@torch.no_grad()
def load_encoders(enc_type, device, debug_mode=False):
    enc_name = enc_type
    encoder_type, architecture, model_config = enc_name.split('-')
    if encoder_type == 'mocov3':
        raise NotImplementedError()

    elif 'dinov2' in encoder_type:
        if 'reg' in encoder_type:
            if debug_mode:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14_reg')
            else:
                encoder = download_load_multiprocess('facebookresearch/dinov2', f'dinov2_vit{model_config}14_reg')
        else:
            if debug_mode:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14')
            else:
                encoder = download_load_multiprocess('facebookresearch/dinov2', f'dinov2_vit{model_config}14')

        del encoder.head
        encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            encoder.pos_embed.data, [16, 16],
        )
        encoder.head = torch.nn.Identity()
        encoder = encoder.to(device)
        encoder.eval()
    
    elif 'dinov1' == encoder_type:
        raise NotImplementedError()

    elif encoder_type == 'clip_dfn':
        # a note for open_clip and timm
        # any model release from timm should not be used in open_clip
        # because it is hardly feasible to manipulate the output
        # for example, if you want the unpooled output instead of the pooled
        # so directly use huggingface for timm models
        import torch.nn.functional as F
        from urllib.request import urlopen
        from PIL import Image
        from open_clip import create_model_from_pretrained, get_tokenizer 

        assert enc_name == "clip_dfn-vit-l/14"
        encoder, preprocess = create_model_from_pretrained(
                                'hf-hub:apple/DFN2B-CLIP-ViT-L-14', 
                                cache_dir="../../models/",  # TODO: better control
                                )
        del encoder.transformer
        visual = encoder.visual
        visual = visual.to(device)
        visual.eval()
        # for output local features
        visual.output_tokens = True
        visual.attn_pool = visual.proj = None

        visual = visual.to(device)
        visual.eval()
        if debug_mode:
            return (visual, preprocess), encoder_type, architecture
        return visual, encoder_type, architecture
    
    elif encoder_type == "siglip":
        # use huggingface api
        from transformers import AutoProcessor, SiglipVisionModel
        from PIL import Image
        import requests

        model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-224").to(device)
        preprocess = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-224")

        assert enc_name == 'siglip-sovit-400m', f'{enc_name} is not supported yet.'
        if debug_mode:
            return (model, preprocess), encoder_type, architecture
        else:
            return model, encoder_type, architecture

    elif encoder_type == 'mae':
        raise NotImplementedError()

    elif encoder_type == 'jepa':
        raise NotImplementedError()

    
    return encoder, encoder_type, architecture

def custom_load(model, state_dict, only_check_missing=True):
    """
    Allow removing the distill_mlp from the state_dict
    """
    if only_check_missing:
        # Load the state_dict with strict=False to ignore unexpected keys
        load_result = model.load_state_dict(state_dict, strict=False)

        # Extract missing and unexpected keys
        missing_keys = load_result.missing_keys
        unexpected_keys = load_result.unexpected_keys

        # Handle missing keys: Raise an error if any keys are missing
        if missing_keys:
            raise KeyError(f"Missing keys in state_dict: {missing_keys}")

        # Optionally, handle unexpected keys (e.g., log them)
        if unexpected_keys:
            print(f"Warning: Unexpected keys in state_dict were ignored: {unexpected_keys}")

        print("Model state_dict loaded successfully with unexpected keys ignored.")
    else:
        model.load_state_dict(state_dict)



if __name__ == "__main__":
    from urllib.request import urlopen
    from PIL import Image
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--test-tok", action="store_true", help="test tokenizer loading function")
    parser.add_argument("--test-dist", action="store_true", help="test distillation teacher model loading function")
    parser.add_argument("--config", type=str, default="configs/vit_s256_c-pt_openmagvit_v2.yaml")
    parser.add_argument("--get-size", action="store_true", help="get the size of the model")
    args = parser.parse_args()

    def _get_param(model):
        return sum(p.numel() for p in model.parameters())
    
    def get_model_size(config):
        model = load_model_from_config(config)
        n_param = sum(p.numel() for p in model.parameters())
        # delete model
        del model
        return n_param
    
    def get_encoder_size(config):
        model = load_model_from_config(config)
        encoder = model.s2to1encoder
        n_param = _get_param(encoder)
        # delete model
        del model
        return n_param
    
    def get_decoder_size(config):
        model = load_model_from_config(config)
        decoder = model.s1to2decoder
        n_param = _get_param(decoder)
        # delete model
        del model
        return n_param
    
    def get_encoder_size_2d(config):
        model = load_model_from_config(config)
        encoder = model.s2dencoder
        n_param = _get_param(encoder)
        # delete model
        del model
        return n_param
    
    def get_decoder_size_2d(config):
        model = load_model_from_config(config)
        decoder = model.s2ddecoder
        n_param = _get_param(decoder)
        # delete model
        del model
        return n_param


    if args.get_size:
        prefix = "configs/v4c/"

        # config_1 = prefix + "VQ_SB256_v4c.yaml"
        # config_2 = prefix + "VQ_BL256_v4c_bsz256.yaml"
        # config_3 = prefix + "VQ_LXL256_v4c_bsz256_ndwm1e.yaml"
        # config_4 = prefix + "VQ_XLXXL256_v4c_bsz256_ndwm1e_entropy5e-3.yaml"
        # configs = [config_1, config_2, config_3, config_4]

        # for config in configs:
        #     if "XXL" not in config:
        #         pass
        #         # print(f"{config}: {get_encoder_size(config):,}")
        #     else:
        #         print(f"Decoder {config}: {get_decoder_size(config):,}")
        #         print(f"Encoder {config}: {get_encoder_size(config):,}")
    
        config_1 = prefix + "VQ_BL256_v4c_bsz256_2d_normal.yaml"
        config_2 = prefix + "VQ_SS256_v4c_2d.yaml"
        config_3 = prefix + "VQ_SB256_v4c_2d_normal.yaml"
        for config in [config_1, config_2, config_3]:
            if "L" not in config:
                print(f"Encoder {config}: {get_encoder_size_2d(config):,}")
            else:
                print(f"Encoder {config}: {get_encoder_size_2d(config):,}")
                print(f"Decoder {config}: {get_decoder_size_2d(config):,}")


    # test loading tokenizer
    if args.test_tok:
        model = load_model_from_config("configs/vit_s256_c-pt_openmagvit_v2.yaml")
        # print the parameters of different components
    
    if args.test_dist:
        # test dinov2
        # distill_encoder, encoder_type, architecture = \
        #     load_encoders("dinov2-vit-g", "cuda:0", debug_mode=True)
        
        # print(distill_encoder.embed_dim)



        # test clip_dfn
        model, encoder_type, architecture = load_encoders(
            "clip_dfn-vit-l/14", 
            device="cuda:0", 
            debug_mode=True)
        visual, preprocess = model

        image = Image.open(urlopen(
            'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
        ))
        image = preprocess(image).unsqueeze(0).to("cuda:0")
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = visual(image)
        
        print(image_features[0].shape)
        print(image_features[1].shape)
        # print(encoder.visual.output_dim)
        print(preprocess)

        # test siglip
        """
        model, encoder_type, architecture = load_encoders("siglip-sovit-400m", device="cuda:0", debug_mode=True)
        print(type(model))
        encoder, processor= model
        image = Image.open(urlopen(
            'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
        ))
        inputs = processor(images=image, return_tensors="pt")
        print(inputs.keys())
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = encoder(**inputs)
            last_hidden_state = outputs.last_hidden_state
        print(last_hidden_state.shape)
        # print(encoder.visual.output_dim)
        print(processor)
        """

