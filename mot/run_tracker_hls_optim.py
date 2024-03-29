import os
import sys
import imageio
import gc
import torch
import numpy as np
from PIL import Image
from yacs.config import CfgNode
import cv2

from mot.deep_sort import preprocessing
from mot.tracklet_processing import save_tracklets, save_tracklets_csv, refine_tracklets, save_tracklets_txt
from mot.tracker import DeepsortTracker, ByteTrackerIOU
from mot.video_output import FileVideo, DisplayVideo, annotate_video_with_tracklets, annotate_video_with_tracklets_iter
from mot.zones import ZoneMatcher
from mot.projection_3d import Projector
from mot.attributes import AttributeExtractorMixed, SpeedEstimator
from evaluate.run_evaluate import run_evaluation

from reid.feature_extractor import FeatureExtractor
from reid.vehicle_reid.load_model import load_model_from_opts

from detection.detection import Detection
from detection.load_detector import load_yolo

from tools.util import FrameRateCounter, Benchmark, Timer, parse_args
from tools.preprocessing import create_extractor
from tools import log
from config.defaults import get_cfg_defaults
from config.config_tools import expand_relative_paths
from config.verify_config import check_mot_config, global_checks

from persistqueue import Queue

MOT_OUTPUT_NAME = "mot"
MAX_NUM_FRAME = 300
NUM_FRAMES_LOOP = 100



def filter_boxes(boxes, scores, classes, good_classes, min_confid=0.5, mask=None):
    """Filter the detected boxes by confidence scores, classes and location.
    Parameters
    ----------
    boxes: list(list)
        Contains [cx, cy, w, h] for each bounding box.
    scores: list(float)
        Confidence scores for each box.
    classes: list(int)
        Class label for each box.
    good_classes: list(int)
        Class labels that we have to keep, and discard others.
    min_confid: float
        Minimal confidence score for a box to be kept.
    mask: Union[None, np.array(np.uint8)]
        A 2d detection mask of zeros and ones. If a point is zero, we discard
        the bounding box whose center lies there, else we keep it.

    Returns
    ------
    final_boxes: list(list)
        The boxes that matched all criteria.
    """
    good_boxes = []
    for bbox, score, cl in zip(boxes, scores, classes):
        if score < min_confid or cl not in good_classes:
            continue
        good_boxes.append(bbox)

    if mask is None:
        return good_boxes

    final_boxes = []
    for bbox in good_boxes:
        cx, cy = int(bbox[0]), int(bbox[1])
        if mask[cy, cx] > 0:
            final_boxes.append(bbox)
    return final_boxes

def box_change_skewed(box, prev_box, skew_ratio=0.1, eps=1e-5):
    """Check if one side of the bounding box has grown a lot more than the opposite one."""
    left_diff = abs(box[0] - prev_box[0])
    right_diff = abs(box[0] + box[2] - (prev_box[0] + prev_box[2]))
    up_diff = abs(box[1] - prev_box[1])
    down_diff = abs(box[1] + box[3] - (prev_box[1] + prev_box[3]))
    lr = max(left_diff, 1) / max(right_diff, 1)
    ud = max(up_diff, 1) / max(down_diff, 1)
    return min(lr, ud) <= skew_ratio or max(lr, ud) >= 1 / skew_ratio


def run_mot(cfg: CfgNode, detector, extractor):
    """Run Multi-object tracking, defined by a config."""

    # check and verify config (has to be done after logging init to see errors)
    if not check_mot_config(cfg):
        return None

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    # free resources
    gc.collect()
    torch.cuda.empty_cache()

    ########################################
    # Loading models, initialization
    ########################################

    # DeepSORT params
    max_cosine_distance = 0.7
    nn_budget = None
    metric = "cosine"

    # non max suppression param
    nms_max_overlap = 0.85
    
    
        
    # newName = name.split('.')[0] + '.avi'
    # videoStreaming =  parent_path+'/'+newName
    # subprocess.call(['ffmpeg', '-i', cfg.MOT.VIDEO, '-c', 'libx265', '-r', '5' , '-vf', "scale=1280:720"  , videoStreaming])

    #video_in = imageio.get_reader("<"+cfg.MOT.VIDEO+">", size=(1280, 720))
    #video_in = imageio.get_reader(cfg.MOT.VIDEO)

    #video_meta = video_in.get_meta_data()
    video_codec = 'mjpeg' 

    #print(video_meta)
    #video_w, video_h = video_meta["size"] #480,640
    #video_w, video_h = 480,640 ; 1280,720
    video_w, video_h = 1280,720
    #video_fps = video_meta["fps"] #30
    video_fps = 30
    VIDEO_EXT = "avi"
    
    # initialize zone matching
    if cfg.MOT.ZONE_MASK_DIR and cfg.MOT.VALID_ZONEPATHS:
        zone_matcher = ZoneMatcher(
            cfg.MOT.ZONE_MASK_DIR, cfg.MOT.VALID_ZONEPATHS)
    else:
        zone_matcher = None

    # initialize 3d projector and speed estimator
    SPEED_WINDOW_SIZE = max(7, round(video_fps / 2.5))
    # minimum area of bounding box to consider for speed calculation
    # about 40x40 in fullHD and 26*26 in HD video
    SPEED_MIN_AREA = int(0.00075 * video_w * video_h)
    projector = Projector(cfg.MOT.CALIBRATION) if cfg.MOT.CALIBRATION else None
    speed_estimator = SpeedEstimator(projector, video_fps) if projector else None

    # initialize tracker
    if cfg.MOT.TRACKER == "deepsort":
        tracker = DeepsortTracker(metric, max_cosine_distance, nn_budget, n_init=3,
                                  zone_matcher=zone_matcher)
        #A lower max_cosine_distance value will result in more 
        #strict matching between features, while a higher value will allow for more flexibility in matching.
        # nn_budget: This parameter controls the maximum number of previous frames that will be used to track 
        # each object. Setting nn_budget to None means that all previous frames will be used for tracking. 
        # However, setting nn_budget to a specific value can help to reduce computational overhead and 
        # improve tracking performance.
        MIN_CONFID = 0.6
    elif cfg.MOT.TRACKER == "bytetrack_iou":
        tracker = ByteTrackerIOU(video_fps, zone_matcher=zone_matcher)
        MIN_CONFID = 0.6
    else:
        raise ValueError("Tracker not implemented.")




    # load input mask if any
    if cfg.MOT.DETECTION_MASK is not None:
        det_mask = Image.open(cfg.MOT.DETECTION_MASK)

        # convert mask to 1's and 0's (with some treshold, because dividing by 255
        # causes some black pixels if the mask is not exactly pixel perfect)
        det_mask = (np.array(det_mask) / 180).astype(np.uint8)

        if len(det_mask.shape) == 3:
            det_mask = det_mask[:, :, 0]

    else:
        det_mask = None

    # initialize output video
    
    if cfg.MOT.ONLINE_VIDEO_OUTPUT:
        video_out = FileVideo(cfg.FONT,
                              os.path.join(cfg.OUTPUT_DIR,
                                           f"{MOT_OUTPUT_NAME}_online_0.{VIDEO_EXT}"),
                              format='FFMPEG', mode='I', fps=video_fps , pixelformat= "yuvj420p",
                              codec=video_codec,
                              #codec="libx264",
                              fontsize=cfg.FONTSIZE)
        
    # initialize display
    if cfg.MOT.SHOW:
        display = DisplayVideo(cfg.FONT)

    ########################################
    # Main tracking loop
    ########################################

    fps_counter = FrameRateCounter()
    benchmark = Benchmark()
    timer = Timer()

    count_save = 0


    cameras_dict = {}
    cameras_dict["0"] = 'rtsp://user-scity01:smartcity01@192.168.30.21/Streaming/channels/101'
    cameras_dict["1"] = 'rtsp://admin:Hik12345@192.168.20.96/Streaming/channels/101'
    cameras_dict["2"] = 'rtsp://admin:Hik12345@192.168.20.175/Streaming/channels/101'

    #Ctic: 
    #cap = cv2.VideoCapture(cameras_dict[str(cfg.MOT.VIDEO)])

    #Local
    cap = cv2.VideoCapture(int(cfg.MOT.VIDEO))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_h)
    
    frame_num = 0
    #for frame_num, frame in enumerate(video_in):
    while True:
        ret, frame = cap.read()
        frame_num+=1
        if frame_num >MAX_NUM_FRAME:
            break
        if cfg.DEBUG_RUN and frame_num >= 80:
            break
        if frame is None or frame.size == 0:
            continue
        frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
        benchmark.restart_timer()
        if frame_num % 1 == 0 :  
            res = detector(frame).xywh[0].cpu().numpy()    
            benchmark.register_call("detector")

            # detected boxes in cx,cy,w,h format
            boxes = [t[:4] for t in res]
            scores = [t[4] for t in res]
            classes = [t[5] for t in res]

            boxes = filter_boxes(boxes, scores, classes,
                                cfg.MOT.TRACKED_CLASSES, MIN_CONFID, det_mask)

            boxes_tlwh = [[int(x - w / 2), int(y - h / 2), w, h]
                        for x, y, w, h in boxes]
            benchmark.register_call("detection filter")

            features = extractor(frame, boxes_tlwh)
            detections = [Detection(bbox, score, clname, feature)
                        for bbox, score, clname, feature in zip(boxes_tlwh, scores, classes, features)]
            features = torch.tensor(features)
            benchmark.register_call("reid")

            boxs = np.array([d.tlwh for d in detections], dtype=int)
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.get_class() for d in detections], dtype=int)

            # run non-maxima supression
            indices = preprocessing.non_max_suppression(
                boxs, classes, nms_max_overlap, scores)
            boxs = [boxs[i] for i in indices]
            scores = [scores[i] for i in indices]
            detections = [detections[i] for i in indices]
            features = features[indices]

            benchmark.register_call("nonmax suppression")

            # get attributes

            benchmark.register_call("attribute extraction")

            # update tracker
            tracker.update(frame_num, detections, {},{})
            benchmark.register_call("tracker")

            active_track_ids = list(tracker.active_track_ids)
            active_tracks = tracker.active_tracks
            active_track_bboxes_tlwh = [tr.bboxes[-1] for tr in active_tracks]

            # estimate speed if possible
            if speed_estimator:
                for track in active_tracks:
                    # only keep bounding boxes that are not cut off / skewed
                    # because those result in inaccurate position approximations:
                    # remaining boxes from the window are stored in last_good_boxes
                    last_good_boxes = []
                    first_frame, last_frame = -1, -1
                    for i in range(len(track.bboxes) - 1, len(track.bboxes) - SPEED_WINDOW_SIZE - 1, -1):
                        if i <= 0:
                            break
                        if not box_change_skewed(track.bboxes[i], track.bboxes[i-1]):
                            last_good_boxes.append(track.bboxes[i])
                            first_frame = track.frames[i]
                            if last_frame < 0:
                                last_frame = first_frame

                    # if there are less than 2 good boxes in the window, cannot estimate speed
                    if len(last_good_boxes) < 2:
                        speed = -1
                    else:
                        last_good_pos = [(round(x[0] + x[2] / 2), x[1] + x[3]) for x in last_good_boxes]
                        speed = speed_estimator.average_speed(last_good_pos, last_frame - first_frame)
                    track.dynamic_attributes.setdefault("speed", []).append(int(speed))

            all_attribs_list = [{} for _ in range(len(active_track_ids))]
            for i, track in enumerate(active_tracks):
                for k, v in track.static_attributes.items():
                    all_attribs_list[i][k] = v[-1]
                for k, v in track.dynamic_attributes.items():
                    all_attribs_list[i][k] = v[-1]

            log.debug(
                f"Frame {frame_num}: active_track_ids: {active_track_ids}, frame type: {type(frame)}, {frame.dtype}, {frame.shape} .")

        if cfg.MOT.ONLINE_VIDEO_OUTPUT: # save without boxes
            video_out.save(frame, active_track_ids,
                                active_track_bboxes_tlwh, all_attribs_list)

        if cfg.MOT.SHOW:
            display.update(frame, active_track_ids,
                            active_track_bboxes_tlwh, all_attribs_list)

        benchmark.register_call("displays")

        fps_counter.step()
            # print("\rFrame: {}/{}, fps: {:.3f}".format(
            #     frame_num, video_frames, fps_counter.value()), end="")
        
        if frame_num> 0 and frame_num % NUM_FRAMES_LOOP == 0 :
            
            # filter unconfirmed tracklets
            final_tracks = list(tracker.tracks.values()) #values es Tracklets del Diccionario (id, Tracklet)
            final_tracks = list(filter(lambda track: len(
                track.frames) >= cfg.MOT.MIN_FRAMES, final_tracks))
            #print(final_tracks)
            #print(len(final_tracks))
            # finalize static attributes and speed
            for track in final_tracks:
                track.predict_final_static_attributes()
                track.finalize_speed()

            log.info("Tracking done. #Tracklets: {}".format(len(final_tracks)))
            # if cfg.MOT.REFINE:
            #     final_tracks = refine_tracklets(final_tracks, zone_matcher)[0]
            #     log.info("Refinement done. #Tracklets remain: {}".format(
            #         len(final_tracks)))

            # compute mean features for tracks and delete frame-by-frame re-id features
            for track in final_tracks:
                track.compute_mean_feature()
                track.features = []

            # csv_save_path = os.path.join(cfg.OUTPUT_DIR, f"{MOT_OUTPUT_NAME}_{count_save}.csv")
            # save_tracklets_csv(final_tracks, csv_save_path)

            # txt_save_path = os.path.join(cfg.OUTPUT_DIR, f"{MOT_OUTPUT_NAME}_{count_save}.txt")
            # save_tracklets_txt(final_tracks, txt_save_path)

            # pkl_save_path = os.path.join(cfg.OUTPUT_DIR, f"{MOT_OUTPUT_NAME}_{count_save}.pkl")
            # save_tracklets(final_tracks, pkl_save_path)

            queue = Queue(os.path.join(cfg.OUTPUT_DIR, f"db"))
            queue.put(final_tracks)
            # Signal the completion of queue creation by creating a flag file
            with open(os.path.join(cfg.OUTPUT_DIR, f"queue_created{count_save}.txt"), 'w') as flag_file:
                flag_file.write("Queue and database created.")

            # if len(cfg.EVAL.GROUND_TRUTHS) == 1:
            #     cfg.defrost()
            #     cfg.EVAL.PREDICTIONS = [txt_save_path]
            #     cfg.freeze()
            #     run_evaluation(cfg)
                    # guardamos videos con nuevos nombres
            video_out.close()
            if cfg.MOT.VIDEO_OUTPUT:
                annotate_video_with_tracklets_iter(os.path.join(cfg.OUTPUT_DIR,
                                                f"{MOT_OUTPUT_NAME}_online_{count_save}.{VIDEO_EXT}"),
                                            os.path.join(cfg.OUTPUT_DIR,
                                                        f"{MOT_OUTPUT_NAME}_{count_save}.{VIDEO_EXT}"),
                                             final_tracks, "yuvj420p", cfg.FONT, cfg.FONTSIZE,count_save, NUM_FRAMES_LOOP)

            count_save += 1 
            for track in final_tracks:
                track.frames.clear()
                track.bboxes.clear()

            # video_out = FileVideo(cfg.FONT,
            #                   os.path.join(cfg.OUTPUT_DIR,
            #                                f"{MOT_OUTPUT_NAME}_online.{VIDEO_EXT}"),
            #                   format='FFMPEG', mode='I', fps=video_meta["fps"],
            #                   codec=video_meta["codec"], 
            #                   pixelformat= "yuvj420p",
            #                   fontsize=cfg.FONTSIZE)

            video_out = FileVideo(cfg.FONT,
                              os.path.join(cfg.OUTPUT_DIR,
                                           f"{MOT_OUTPUT_NAME}_online_{count_save}.{VIDEO_EXT}"),
                              format='FFMPEG', mode='I', fps=video_fps , pixelformat= "yuvj420p",
                              codec=video_codec,
                              fontsize=cfg.FONTSIZE)

            

    #video_in.close()
    time_taken = f"{int(timer.elapsed() / 60)} min {int(timer.elapsed() % 60)} sec"
    # avg_fps = video_frames / timer.elapsed()
    # log.info(
    #     f"\nTracking finished over {video_frames} frames, total time: {time_taken}, average fps: {avg_fps:.3f}.")
    log.info(f"MOT Benchmark (times in ms)\n{benchmark.get_benchmark()}")

    ########################################
    # Run postprocessing and save results
    ########################################

    if cfg.MOT.SHOW:
        display.close()

    if cfg.MOT.ONLINE_VIDEO_OUTPUT:
        video_out.close()

    ### here pass into loop
    #return 0
    return 0


if __name__ == "__main__":
    args = parse_args("Run Multi-object tracker on a video.")
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

    run_mot(cfg)
