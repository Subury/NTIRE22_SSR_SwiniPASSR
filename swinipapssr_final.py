from pyexpat import model
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None, help='model path.')
parser.add_argument('--iter', type=int, default=0, help='average iter number.')
args = parser.parse_args()

data = {}
models = [torch.load(f"{args.path}/models/{index}_E.pth", map_location="cpu") for index in range(5000, args.iter + 1, 5000)]
models += [torch.load(f"{args.path}/models/{index}_G.pth", map_location="cpu") for index in range(5000, args.iter + 1, 5000)]

for key in models[-1].keys():
    data[key] = sum([model[key] for model in models]) / len(models)

data["upsample.0.weight"] = data.pop("upsample_middle.0.weight")
data["upsample.0.bias"] = data.pop("upsample_middle.0.bias")
data["upsample.2.weight"] = data.pop("upsample_last.0.weight")
data["upsample.2.bias"] = data.pop("upsample_last.0.bias") 

temp1, temp2 = data.pop("conv_middle.weight"), data.pop("conv_middle.bias")

torch.save(data, "./pretrained/swinipapssr_final.pth")