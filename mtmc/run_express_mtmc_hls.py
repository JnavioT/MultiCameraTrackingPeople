import sys
import os
import time

from yacs.config import CfgNode
import subprocess

from mot.run_tracker_hls import run_mot, MOT_OUTPUT_NAME
from mtmc.run_mtmc import run_mtmc
from mtmc.output import save_tracklets_per_cam, save_tracklets_csv_per_cam, save_tracklets_txt_per_cam, annotate_video_mtmc, annotate_video_mtmc_iter
from evaluate.run_evaluate import run_evaluation
from config.defaults import get_cfg_defaults
from config.config_tools import expand_relative_paths
from config.verify_config import check_express_config, global_checks, check_mot_config
from tools.util import parse_args
from tools import log

import threading
import concurrent.futures

MTMC_OUTPUT_NAME = "mtmc"

def run_express_mtmc(cfg: CfgNode):
    """Run Express MTMC on a given config."""
    if not check_express_config(cfg):
        return None
    mot_configs = []
    cam_names, cam_dirs = [], []
    for cam_idx, cam_info in enumerate(cfg.EXPRESS.CAMERAS): # x cada camara
        cam_cfg = cfg.clone()
        cam_cfg.defrost()
        for key, val in cam_info.items():
            #print(key, '-', val)
            setattr(cam_cfg.MOT, key.upper(), val)
        #print(cam_cfg.MOT.VIDEO) # C:\Users\JOSE\CTIC\vehicle_mtmc\<video1>
        #cam_video_name = os.path.split(cam_cfg.MOT.VIDEO)[1].split(".")[0] # '<video0>'
        cam_video_name = cam_cfg.MOT.VIDEO
        cam_names.append(cam_video_name) # ['<video0>', '<video1>']
        #print(cam_names)
        # set output dir of MOT to a unique folder under the root OUTPUT_DIR
        cam_dir = os.path.join(cfg.OUTPUT_DIR, f"{cam_idx}_{cam_video_name}") #0_v1_hls , 0_trasero, 1_entrada
        cam_dirs.append(cam_dir) # donde se guardaran los mcmt y mot
        #print(cam_dirs) 
        #['C:\\Users\\JOSE\\CTIC\\vehicle_mtmc\\output\\webcam_02_cameras\\0_video0', 
        # 'C:\\Users\\JOSE\\CTIC\\vehicle_mtmc\\output\\webcam_02_cameras\\1_video1']
        cam_cfg.OUTPUT_DIR = cam_dir ######-> 'C:\\Users\\JOSE\\CTIC\\vehicle_mtmc\\output\\webcam_02_cameras\\0_video0'
        if len(cfg.EVAL.GROUND_TRUTHS) == len(cfg.EXPRESS.CAMERAS):
            cam_cfg.EVAL.GROUND_TRUTHS = [cfg.EVAL.GROUND_TRUTHS[cam_idx]]
        cam_cfg.freeze()

        mot_configs.append(cam_cfg) #OUTPUT_DIR: "output/ctic_hls_test1" + /0_v1_hls
        #mot_configs.append()
        if not check_mot_config(cam_cfg):
            log.error(
                f"Error in the express config of camera {len(mot_configs) - 1}.")
            return None

    # run MOT in all cameras
    # for mot_conf in mot_configs:
    #     run_mot(mot_conf) # hace correr run_mot en cada config con cada salida

    def loop1():
        run_mot(mot_configs[0])

    def loop2():
        run_mot(mot_configs[1])


    log.info("Express: Running MOT on all cameras finished. Running MTMC...")

    # run MTMC
    # secuencia run_mot, cada camara -> para cada particion (2,3..,n) (mot_0.pkl, mot_1.pkl, mot_2.pkl ) calcula
    # calcula mtmc para cada .pkl que salió de correr run_mot

    def loop3():
        i = 0
        #for i in range(8): # cambiar por un True : si existe file mot_0.pkl : aumenta a mot_1.pkl
        while True:
            if i>=3: 
                break
            while not (os.path.isfile(os.path.join(cam_dirs[1],f"{MOT_OUTPUT_NAME}_{i}.pkl"))
                and os.path.isfile(os.path.join(cam_dirs[0],f"{MOT_OUTPUT_NAME}_{i}.pkl"))):
                time.sleep(1)
            pickle_paths = [os.path.join(
                path, f"{MOT_OUTPUT_NAME}_{i}.pkl") for path in cam_dirs]
            mtmc_cfg = cfg.clone()
            mtmc_cfg.defrost()
            mtmc_cfg.MTMC.PICKLED_TRACKLETS = pickle_paths
            mtmc_cfg.freeze()
            mtracks = run_mtmc(mtmc_cfg) # se puede paralelizar para cada camara pq lo hace serial

            log.info("Express: Running MTMC on all cameras finished. Saving final results ...")

            # save single cam tracks
            final_pickle_paths = [os.path.join(
                d, f"{MTMC_OUTPUT_NAME}_{i}.pkl") for d in cam_dirs]
            # final_csv_paths = [os.path.join(
            #     d, f"{MTMC_OUTPUT_NAME}_{i}.csv") for d in cam_dirs]
            # final_txt_paths = [os.path.join(
            #     d, f"{MTMC_OUTPUT_NAME}_{i}.txt") for d in cam_dirs]
            save_tracklets_per_cam(mtracks, final_pickle_paths)
            # save_tracklets_txt_per_cam(mtracks, final_txt_paths)
            # save_tracklets_csv_per_cam(mtracks, final_csv_paths)
            
            if cfg.EXPRESS.FINAL_VIDEO_OUTPUT:
                for j, cam_dir in enumerate(cam_dirs): # lo hace serial pero podria ser paralelo
                        #['C:\\Users\\JOSE\\CTIC\\vehicle_mtmc\\output\\webcam_02_cameras\\0_video0', 
                    # 'C:\\Users\\JOSE\\CTIC\\vehicle_mtmc\\output\\webcam_02_cameras\\1_video1']
                        #video_in = mot_configs[i].MOT.VIDEO
                    video_in = os.path.join(
                    cam_dir, f"mot_online_{i}.avi")

                    #video_ext = video_in.split(".")[1]
                    video_ext = "avi"
                    video_out = os.path.join(
                        cam_dir, f"{MTMC_OUTPUT_NAME}_{j}_{i}.{video_ext}")
                    ouput_hls = os.path.join(
                        cam_dir, f"hls_{j}_{i}.m3u8")
                    annotate_video_mtmc_iter(video_in, video_out, mtracks,
                                            j, "yuvj420p",i,100,font=cfg.FONT, fontsize=cfg.FONTSIZE)
                    #mtmc_0_0
                    subprocess.call(['ffmpeg', '-i',  video_out, '-c', 'libx264', '-preset', 'slow' , \
                     '-b', '500k', '-b', '128k', '-f', 'hls' ,  \
                     '-hls_list_size', '0','-hls_time', '2', ouput_hls])
                    log.info(f"Express: video cam{j}_iter{i} saved.")
            i+=1
        
    
    
    # Create two threads, one for each loop
    thread1 = threading.Thread(target=loop1)
    thread2 = threading.Thread(target=loop2)
    #thread3 = threading.Thread(target=loop3)
    future = concurrent.futures.Future()
    t = threading.Thread(target=lambda: future.set_result(loop3()))

    # Start both threads
    thread1.start()
    thread2.start()
    t.start()
    # Wait for both threads to finish
    thread1.join()
    thread2.join()
    t.join()
    mtracks = future.result()

    # if cfg.EXPRESS.FINAL_VIDEO_OUTPUT:
    #     for j, cam_dir in enumerate(cam_dirs): # lo hace serial pero podria ser paralelo
    #             #['C:\\Users\\JOSE\\CTIC\\vehicle_mtmc\\output\\webcam_02_cameras\\0_video0', 
    #         # 'C:\\Users\\JOSE\\CTIC\\vehicle_mtmc\\output\\webcam_02_cameras\\1_video1']
    #             #video_in = mot_configs[i].MOT.VIDEO
    #         video_in = os.path.join(
    #         cam_dir, "mot_online.avi")

    #         #video_ext = video_in.split(".")[1]
    #         video_ext = "avi"
    #         video_out = os.path.join(
    #             cam_dir, f"{MTMC_OUTPUT_NAME}_{j}.{video_ext}")
    #         annotate_video_mtmc(video_in, video_out, mtracks,
    #                                 j, "yuvj420p",font=cfg.FONT, fontsize=cfg.FONTSIZE)
    #         log.info(f"Express: video {j} saved.")

    # if len(cfg.EVAL.GROUND_TRUTHS) == 0:
    #     log.info("Ground truths are not provided for evaluation, terminating.")
    #     return mtracks

    # log.info("Ground truth annotations are provided, trying to evaluate MTMC ...")
    # if len(cfg.EVAL.GROUND_TRUTHS) != len(cam_names):
    #     log.error(
    #         "Number of ground truth files != number of cameras, aborting evaluation ...")
    #     return mtracks

    # mtmc_cfg.defrost()
    # mtmc_cfg.EVAL.PREDICTIONS = final_txt_paths
    # mtmc_cfg.freeze()
    # eval_res = run_evaluation(mtmc_cfg)

    # if eval_res:
    #     log.info("Evaluation successful.")
    # else:
    #     log.error("Evaluation unsuccessful: probably EVAL config had some errors.")

    #return mtracks
    return 0


if __name__ == "__main__":
    args = parse_args("Express MTMC: run MOT on all cameras and then MTMC.")
    cfg = get_cfg_defaults()

    if args.config:
        cfg.merge_from_file(os.path.join(cfg.SYSTEM.CFG_DIR, args.config))
    cfg = expand_relative_paths(cfg)
    cfg.freeze()

    # initialize output directory and logging
    if not global_checks["OUTPUT_DIR"](cfg.OUTPUT_DIR):
        log.error(
            "Invalid param value in: OUTPUT_DIR. Provide an absolute path to a directory, whose parent exists.")
        sys.exit(2)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    log.log_init(os.path.join(cfg.OUTPUT_DIR, args.log_filename),
                 args.log_level, not args.no_log_stdout)
    run_express_mtmc(cfg)
