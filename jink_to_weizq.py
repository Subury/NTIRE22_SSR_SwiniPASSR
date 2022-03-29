from xxlimited import new
import torch

data = torch.load("./final_models/P48W12D12E180H10.pth", map_location="cpu")['state_dict']

for key in list(data.keys()):
    
    new_key_name = ".".join(key.split('.')[2:])

    if 'layers.3' in new_key_name:
        new_key_name = new_key_name.replace('layers.3', 'second_layers.0')

    elif 'layers.4' in new_key_name:
        new_key_name = new_key_name.replace('layers.4', 'second_layers.1')

    elif 'layers.5' in new_key_name:
        new_key_name = new_key_name.replace('layers.5', 'second_layers.2')
    
    elif 'pam_conv' in new_key_name:
        new_key_name = new_key_name.replace('pam_conv', 'conv_after_body')

    elif 'conv_after_body' in new_key_name:
        new_key_name = new_key_name.replace('conv_after_body', 'second_conv_after_body')

    data[new_key_name] = data.pop(key)


# for key in data.keys():
#     print(key)

torch.save(data, "./final_models/P48W12D12E180H10.pth")