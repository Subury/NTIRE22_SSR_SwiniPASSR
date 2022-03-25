from pyexpat import model
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--iter', type=int, default=0, help='average iter number.')
args = parser.parse_args()

data = {}
models = [torch.load(f"./superresolution/ssr_flickr1024_swinipassr_plus/models/{index}_E.pth", map_location="cpu") for index in range(5000, args.iter + 1, 5000)]
models += [torch.load(f"./superresolution/ssr_flickr1024_swinipassr_plus/models/{index}_G.pth", map_location="cpu") for index in range(5000, args.iter + 1, 5000)]

for key in models[-1].keys():
    data[key] = sum([model[key] for model in models]) / len(models)

torch.save(data, "./pretrained/swinipassr_final.pth")