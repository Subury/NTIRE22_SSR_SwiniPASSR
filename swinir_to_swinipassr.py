
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--swinir_path', type=str, default=None, help='Path to swinir file.')
parser.add_argument('--swinipassr_path', type=str, default=None, help='Path to swinipassr file.')
args = parser.parse_args()

data = torch.load("./superresolution/sr_flickr1024_swinir/models/500000_G.pth ", map_location="cpu")
new_data = {}

for key in list(data.keys()):
    
    if ("conv_first" in key) or ("patch_embed" in key):
        new_data[key] = data.pop(key)
    
    elif "layers.0" in key:
        new_data[key] = data.pop(key)

    elif "layers.1" in key:
        new_data[key] = data.pop(key)

    elif "layers.2" in key:
        new_data[key] = data.pop(key)

    elif "layers.3" in key:
        new_data[key.replace("layers.3", "second_layers.0")] = data.pop(key)

    elif "layers.4" in key:
        new_data[key.replace("layers.4", "second_layers.1")] = data.pop(key)

    elif "layers.5" in key:
        new_data[key.replace("layers.5", "second_layers.2")] = data.pop(key)
    
    elif ("conv_before_upsample" in key) or ("upsample" in key) or ("conv_last" in key):
        new_data[key] = data.pop(key)
    
    elif key == "conv_after_body.weight":
        new_data["second_" + key] = data.pop(key)
    
    elif key == "conv_after_body.bias":
        new_data["second_" + key] = data.pop(key)

    elif key == "norm.weight":
        new_data["second_" + key] = data.pop(key)

    elif key == "norm.bias":
        new_data["second_" + key] = data.pop(key)
    else:
        print(key)
    
torch.save(new_data, "/pretrained/pre_swinipassr.pth")
    
