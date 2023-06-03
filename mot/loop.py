import sys
import os
import imageio
import imageio.v3 as iio
import imageio.v2 as iio2
import numpy as np


from yacs.config import CfgNode

from mot.run_tracker import run_mot, MOT_OUTPUT_NAME


output_path1 = "datasets/camera_creating/myvideo.avi"
output_path2 = "datasets/camera_creating/myvideo2.avi"
fps = 60
codec = "libx265"
output_writer1 = iio2.get_writer(output_path1,format='FFMPEG', mode='I', fps=fps, codec=codec)
output_writer2 = iio2.get_writer(output_path2,format='FFMPEG', mode='I', fps=fps, codec=codec)

#stream = imageio.get_reader('<video0>', 'video4linux2', mode='I', fps=30)

for idx, frame in enumerate(iio.imiter("<video0>")):
#for idx, frame in enumerate(iio.imiter(video_url)): ## convertir en streaming # desde youtube, webcam "<video0>"
    print(f"Frame {idx}: avg. color {np.sum(frame, axis=-1)}")
    if idx < 180:
        output_writer1.append_data(frame)
    elif idx == 180:
        output_writer1.close()
    elif idx >180 and idx <360:
        output_writer2.append_data(frame)
    elif idx == 360:
        output_writer2.close()
        break

### loop en la carpeta run_tracker

#main():

# thread1: loop camara:
    #guarda videos cada 10 segundos y los hace pasar por run_tracker
# thread2: loop camara:
    #guarda videos cada 10 segundos y los hace pasar por run_tracker
# espera a que finalice thread 1 y 2
    # comparte resultados con mtmc para asignar ids.
    # guarda videos
    