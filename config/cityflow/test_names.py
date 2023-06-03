import os
dir = "datasets/hls_ctic/video1/v1_hls.m3u8"
print(os.path.split(dir))
name = os.path.split(dir)[1].split(".")[0]
print(name)