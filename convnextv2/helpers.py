import torch
from collections import OrderedDict
import math


def remap_checkpoint_keys(ckpt): 
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():   #改预训练模型中一些参数的名字，因为convnext_unet里没有encoder等等这些名字
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
    for k in ["head.weight", "head.bias"]:  #去掉加载的模型的头
        if (
            k in checkpoint_model
            and checkpoint_model[k].shape != state_dict[k].shape
        ):
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # remove decoder weights
    checkpoint_model_keys = list(checkpoint_model.keys())
    for k in checkpoint_model_keys:
        if "decoder" in k or "mask_token" in k or "proj" in k or "pred" in k:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    checkpoint_model = remap_checkpoint_keys(checkpoint_model)
    load_state_dict(model, checkpoint_model, prefix="")


    # print("---冻结编码器---")
    # for layer, layer2 in zip(model.initial_conv, model.initial_conv2):
    #     for param, param2 in zip(layer.parameters(), layer2.parameters()):
    #         param.requires_grad = False
    #         param2.requires_grad = False

    # for layer, layer2 in zip(model.stem, model.stem2):
    #     for param, param2 in zip(layer.parameters(), layer2.parameters()):
    #         param.requires_grad = False
    #         param2.requires_grad = False

    # for layer, layer2 in zip(model.stages, model.stages2):
    #     for param, param2 in zip(layer.parameters(), layer2.parameters()):
    #         param.requires_grad = False
    #         param2.requires_grad = False

    # for layer, layer2 in zip(model.downsample_layers, model.downsample_layers2):
    #     for param, param2 in zip(layer.parameters(), layer2.parameters()):
    #         param.requires_grad = False
    #         param2.requires_grad = False

    return model
