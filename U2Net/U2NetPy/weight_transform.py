import os
import torch

weight_path = "/raid/home/guiyan/bingqiangzhou/projects/sod/codes/saved_models/u2net.pth"

groups = 2

state_dict = torch.load(weight_path)
for key in state_dict.keys():
    print(key, " weights before shape: ", state_dict[key].shape)
    if 'stage' in key:
        if "stage1" in key and "rebnconvin" in key:
            continue
        elif 'rebnconv' in key and 'conv_s1' in key:
            chunk_dim = 1
            if "bias" in key:
                # chunk_dim = 0
                continue
            weights = state_dict[key]
            new_weight = []
            for channel in torch.chunk(weights, weights.shape[chunk_dim] // groups, dim=chunk_dim):
                new_weight.append(torch.mean(channel, dim=chunk_dim, keepdim=True))
            new_weight = torch.cat(new_weight, dim=chunk_dim)
            state_dict.update({key: new_weight})

    print(key, " weights after shape: ", state_dict[key].shape)

model_path_with_name, ext = os.path.splitext(weight_path)
new_model_path = f"{model_path_with_name}_groups{groups}{ext}"
torch.save(state_dict, new_model_path)
print(f"modify model file save to: {new_model_path}")
