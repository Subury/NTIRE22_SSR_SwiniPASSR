import torch

data = {}

G0 = torch.load("./superresolution/ssr_flickr1024_swinipassr/models/480000_G.pth", map_location='cpu')
G1 = torch.load("./superresolution/ssr_flickr1024_swinipassr/models/485000_G.pth", map_location='cpu')
G2 = torch.load("./superresolution/ssr_flickr1024_swinipassr/models/490000_G.pth", map_location='cpu')
G3 = torch.load("./superresolution/ssr_flickr1024_swinipassr/models/495000_G.pth", map_location='cpu')
G4 = torch.load("./superresolution/ssr_flickr1024_swinipassr/models/500000_G.pth", map_location='cpu')

E0 = torch.load("./superresolution/ssr_flickr1024_swinipassr/models/480000_E.pth", map_location='cpu')
E1 = torch.load("./superresolution/ssr_flickr1024_swinipassr/models/485000_E.pth", map_location='cpu')
E2 = torch.load("./superresolution/ssr_flickr1024_swinipassr/models/490000_E.pth", map_location='cpu')
E3 = torch.load("./superresolution/ssr_flickr1024_swinipassr/models/495000_E.pth", map_location='cpu')
E4 = torch.load("./superresolution/ssr_flickr1024_swinipassr/models/500000_E.pth", map_location='cpu')

for key in G0.keys():

    data[key] = (G0[key] + G1[key] + G2[key] + G3[key] + G4[key] + E0[key] + E1[key] + E2[key] + E3[key] + E4[key]) / 10.0

torch.save(data, './pretrained/pre_swinipassr_plus.pth')
