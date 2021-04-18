import os
import torch



def weight_transfrom_for_groupconv(weight_path, groups=2):

    state_dict = torch.load(weight_path)
    for key in state_dict.keys():
        print(key, " weights before shape: ", state_dict[key].shape)
        if 'stage' in key:
            if "stage1" in key and "rebnconvin" in key:
                continue
            elif 'rebnconv' in key and 'conv_s1' in key and 'weight' in key:
                weights = state_dict[key]
                new_weight = []
                for channel in torch.chunk(weights, weights.shape[1] // groups, dim=1):
                    new_weight.append(torch.mean(channel, dim=1, keepdim=True))
                new_weight = torch.cat(new_weight, dim=1)
                state_dict.update({key: new_weight})

        print(key, " weights after shape: ", state_dict[key].shape)
    model_path_with_name, ext = os.path.splitext(weight_path)
    new_model_path = f"{model_path_with_name}_groups{groups}{ext}"
    torch.save(state_dict, new_model_path)
    print(f"modify model file save to: {new_model_path}")

def weight_transfrom_for_dsconv(weight_path, groups=2):

    state_dict = torch.load(weight_path)
    for key in list(state_dict.keys()):
        print(key, "old weights shape: ", state_dict[key].shape)
        if 'stage' in key:
            if 'rebnconv' in key and 'conv_s1' in key:
                if "bias" in key:
                    # 1.bias: only need rename
                    new_key = key.replace('bias', '1.bias')
                    state_dict.update({new_key: state_dict[key]})
                    print(new_key, " weights shape: ", state_dict[new_key].shape)
                    del state_dict[key]

                if 'weight' in key:
                    weights = state_dict[key]
                    
                    # 0.weight： [out_channels, in_channels, 3, 3] -> [in_channels, 1, 3, 3]
                    new_weight = torch.mean(state_dict[key], dim=0, keepdim=True).transpose(0, 1)
                    new_key = key.replace('weight', '0.weight')
                    state_dict.update({new_key: new_weight})
                    print(new_key, " weights shape: ", state_dict[new_key].shape)
                    
                    # 0.bias： [in_channels]
                    new_weight = torch.zeros(weights.shape[1])
                    new_key = key.replace('weight', '0.bias')
                    state_dict.update({new_key: new_weight})
                    print(new_key, " weights shape: ", state_dict[new_key].shape)

                    # 1.weight: [out_channels, in_channels, 3, 3] -> [out_channels, in_channels, 1, 1]
                    new_weight = torch.mean(torch.mean(state_dict[key], dim=2, keepdim=True), dim=3, keepdim=True)
                    new_key = key.replace('weight', '1.weight')
                    state_dict.update({new_key: new_weight})
                    print(new_key, " weights shape: ", state_dict[new_key].shape)

                    # remove old weight
                    del state_dict[key]

    # print(key, " weights after shape: ", state_dict[key].shape)
    model_path_with_name, ext = os.path.splitext(weight_path)
    new_model_path = f"{model_path_with_name}_dsconv{ext}"
    torch.save(state_dict, new_model_path)
    print(f"modify model file save to: {new_model_path}")

weight_path = "/raid/home/guiyan/bingqiangzhou/projects/sod/codes/saved_models/u2net.pth"
# weight_transfrom_for_groupconv(weight_path, groups=2)
weight_transfrom_for_dsconv(weight_path)