import re
import matplotlib.pyplot as plt
import glob


def extract_txt_content(txt_file_path):
    pattern = r"train\ loss:\ [0-9.]+,\ tar:\ [0-9.]+"
    prog = re.compile(pattern)

    result_list = {
        "train loss": [],
        "tar": []
    }

    with open(txt_file_path, 'r') as f:
        line = f.readline()
        i = 1
        while(len(line) > 0): 
            m = prog.search(line)
            if m is not None:
                search_result = line[m.start():m.end()]
                result_dict = search_result.split(",")
                for result in result_dict:
                    name, value = result.split(':')
                    result_list[name.strip()].append(float(value.strip()))
                # print(line + "line end", len(line), " search result:", line[m.start():m.end()])

            line = f.readline()
            i += 1

    for key in result_list.keys():
        plt.plot(result_list[key], label=key)
    plt.legend()
    plt.title("trainloss b16 2gpu pretrain")
    plt.savefig("iter.png")

def extract_content_form_model_names(file_dir, gap=2000):
    file_path = glob.glob(file_dir+"*.pth")

    pattern = r"[0-9.]+_train_[0-9.]+_tar_[0-9.]+.pth"
    prog = re.compile(pattern)

    result_list = {
        "train": [0.0 for i in range(1000)],
        "tar": [0.0 for i in range(1000)]
    }

    for path in file_path:
        m = prog.search(path)
        search_result = path[m.start(): m.end()-4]
        # print(search_result)

        results = search_result.split('_')
        index = int(results[0]) // gap - 1
        result_list[results[1]][index] = float(results[2])
        result_list[results[3]][index] = float(results[4])
    cols = len(result_list.keys())
    plt.figure(figsize=(cols*5, 4))
    for i, key in enumerate(result_list.keys()):
        plt.subplot(1, cols, i+1)
        index = result_list[key].index(0.0)
        plt.plot(result_list[key][:index], label=key)
        plt.legend()
        plt.xlabel(f"{gap} iter/ times")
        plt.ylabel(f"{key} loss")
    plt.suptitle(f"trainloss b16 2gpu pretrain")
    plt.savefig(f"iter_pre_{gap}.png")


txt_file_path = "../dsconv.out"
extract_txt_content(txt_file_path)

# file_dir = "../models/u2net/"
# extract_content_form_model_names(file_dir)