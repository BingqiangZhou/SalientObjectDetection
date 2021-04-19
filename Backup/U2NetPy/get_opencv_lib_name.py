from glob import glob
import os

lib_dir = "D:/OpenCV4/opencv_cuda_build/install/x64/vc15/lib/"
lib_version = '451'
lib_suffix = {
    "debug" : f"*{lib_version}d.lib", 
    "release" : f"*{lib_version}.lib"
}
for suffix in lib_suffix.keys():
    print(suffix, lib_suffix[suffix], sep="\n\n\n")
    for path in glob(lib_dir + lib_suffix[suffix]):
        print(os.path.split(path)[-1])