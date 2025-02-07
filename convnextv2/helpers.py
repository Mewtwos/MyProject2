import re
import torch
from collections import OrderedDict
import math


def remap_checkpoint_keys(ckpt): 
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():  
        if "bn" in k:
            k = k.replace(".bn", "")
        if k.startswith("encoder2"):
            k = k.replace("encoder2", "Encoder2")
        if k.startswith("encoder"):
            k = ".".join(k.split(".")[1:])  # remove encoder in the name
        if k.endswith("kernel"):
            k = ".".join(k.split(".")[:-1])  # remove kernel in the name
            new_k = k + ".weight"
            if len(v.shape) == 3:  # resahpe standard convolution
                kv, in_dim, out_dim = v.shape
                ks = int(math.sqrt(kv))
                new_ckpt[new_k] = (
                    v.permute(2, 1, 0).reshape(out_dim, in_dim, ks, ks).transpose(3, 2)
                )
            elif len(v.shape) == 2:  # reshape depthwise convolution
                kv, dim = v.shape
                ks = int(math.sqrt(kv))
                new_ckpt[new_k] = (
                    v.permute(1, 0).reshape(dim, 1, ks, ks).transpose(3, 2)
                )
            continue
        elif "ln" in k or "linear" in k:
            k = k.split(".")
            k.pop(-2)  # remove ln and linear in the name
            new_k = ".".join(k)
        elif "backbone.resnet" in k:
            # sometimes the resnet model is saved with the prefix backbone.resnet
            # we need to remove this prefix
            new_k = k.split("backbone.resnet.")[1]
        else:
            new_k = k
        new_ckpt[new_k] = v
    #为第二个编码器分支创建key
    for k, v in new_ckpt.copy().items():
        if "downsample_layers" in k:
            new_k = k.replace("downsample_layers", "downsample_layers2")
            new_ckpt[new_k] = v
        if "initial_conv" in k:
            new_k = k.replace("initial_conv", "initial_conv2")
            new_ckpt[new_k] = v
        if "stem" in k:
            new_k = k.replace("stem", "stem2")
            new_ckpt[new_k] = v
        if "stages" in k:
            new_k = k.replace("stages", "stages2")
            new_ckpt[new_k] = v
    # for k, v in new_ckpt.copy().items():
    #     if "Encoder2" in k:
    #         oldk = k
    #         k = ".".join(k.split(".")[1:])  # remove encoder in the name
            # del new_ckpt[oldk]
            # if "downsample_layers" in k:
            #     new_k = k.replace("downsample_layers", "downsample_layers2")
            #     new_ckpt[new_k] = v
            # if "initial_conv" in k:
            #     new_k = k.replace("initial_conv", "initial_conv2")
            #     new_ckpt[new_k] = v
            # if "stem" in k:
            #     new_k = k.replace("stem", "stem2")
            #     new_ckpt[new_k] = v
            # if "stages" in k:
            #     new_k = k.replace("stages", "stages2")
            #     new_ckpt[new_k] = v
    # reshape grn affine parameters and biases
    for k, v in new_ckpt.items():
        if k.endswith("bias") and len(v.shape) != 1:
            new_ckpt[k] = v.reshape(-1)
        elif "grn" in k:
            new_ckpt[k] = v.unsqueeze(0).unsqueeze(1)
    return new_ckpt


def load_state_dict(
    model, state_dict, prefix="", ignore_missing="relative_position_index", quit=False
):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split("|"):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print(
            "Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys
            )
        )
    if len(unexpected_keys) > 0:
        print(
            "Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys
            )
        )
    if len(ignore_missing_keys) > 0:
        print(
            "Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys
            )
        )
    if len(error_msgs) > 0:
        print("\n".join(error_msgs))


def load_custom_checkpoint(model, pretrained_path):
    checkpoint = torch.load(pretrained_path, map_location="cpu")#加载预训练模型

    print("Load pre-trained checkpoint from: %s" % pretrained_path)
    checkpoint_model = checkpoint["model"] if "model" in checkpoint else checkpoint
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if (
            k in checkpoint_model
            and checkpoint_model[k].shape != state_dict[k].shape
        ):
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # remove decoder weights
    checkpoint_model_keys = list(checkpoint_model.keys())
    for k in checkpoint_model_keys:
        if "decoder" in k or "mask_token" in k or "pred" in k or "loss_fn" in k or "dwtaf" in k or "proj" in k or "encoder2" in k or "bn" in k: #dwtaf proj
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    checkpoint_model = remap_checkpoint_keys(checkpoint_model)
    load_state_dict(model, checkpoint_model, prefix="")

    return model

def modify_string(input_str):
    # 使用正则表达式匹配 stages.n.n.xxx.xxx 格式的字符串
    pattern = r"^(stages\.\d\.\d\.)"

    # 替换为 "stages.n.n.blockx" 的形式
    modified_str1 = re.sub(pattern, r"\1blockx.", input_str)
    modified_str2 = re.sub(pattern, r"\1blocky.", input_str)

    return modified_str1, modified_str2

def load_imagenet_checkpoint(model, pretrained_model):
    checkpoint = torch.load(pretrained_model, map_location="cpu")#加载预训练模型

    print("Load pre-trained checkpoint from: %s" % pretrained_model)
    checkpoint_model = checkpoint["model"] if "model" in checkpoint else checkpoint

    #remap keys
    new_ckpt = OrderedDict()
    for k, v in checkpoint_model.items():
        if "downsample_layers" in k:
            new_ckpt[k] = v
            new_k = k.replace("downsample_layers", "downsample_layers2")
            new_ckpt[new_k] = v
            continue
        if "stages" in k:
            k1, k2 = modify_string(k)
            new_ckpt[k1] = v
            new_ckpt[k2] = v
            continue
    for k, v in new_ckpt.items():
        if "grn" in k:
            new_ckpt[k] = v.unsqueeze(0).unsqueeze(1)
    load_state_dict(model, new_ckpt, prefix="")
    
    return model
