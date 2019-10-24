#! /usr/bin/env python

import os
import cv2
import numpy as np
from tqdm import tqdm
from utils import draw_boxes
from frontend import YOLO
import json
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def _main_():
    config_path = 'config.json'
    weights_path = 'model.h5'

    # image_paths = [
    #     "/media/wd-4tb-hdd/data/experiments/20190517_prototype_steel_mma/12-0001-05172019190000.avi",
    #     "/media/wd-4tb-hdd/data/experiments/20190517_prototype_steel_mma/12-0002-05172019185958.avi",
    #     "/media/wd-4tb-hdd/data/experiments/20190517_prototype_steel_mma/12-0003-05172019185959.avi"
    # ]
    root_dir = "/media/wd-4tb-hdd/data/experiments/20190517_prototype_steel_mma"
    image_paths = [
        "2-0001-05172019184150.avi",
        "2-0002-05172019184149.avi",
        "2-0003-05172019184148.avi",
    ]
    image_paths = [os.path.join(root_dir, p) for p in image_paths]

    show_debug_frame = True
    save_output_video = False
    write_raw_results_json = False
    num_frames_to_save = 30  # set to None in order to run all frames in each video
    conf_threshold = 0.20

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(backend=config['model']['backend'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################
    for image_path in image_paths:
        if image_path[-4:] == '.avi':
            detection_dict = {}
            video_out = image_path[:-4] + '_detected' + image_path[-4:]
            video_reader = cv2.VideoCapture(image_path)

            nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

            if save_output_video:
                video_writer = cv2.VideoWriter(video_out,
                                               cv2.VideoWriter_fourcc(*'MPEG'),
                                               50.0,
                                               (frame_w, frame_h))

            total_predict_time_s = 0
            num_frames_to_run = nb_frames if num_frames_to_save is None else num_frames_to_save
            for frame_i in tqdm(range(num_frames_to_run)):
                _, image = video_reader.read()
                start_t = time.time()
                boxes = yolo.predict(image, conf_threshold=conf_threshold)
                predict_t = time.time() - start_t
                total_predict_time_s += predict_t
                # print("Predict time: {:.5f}s".format(predict_t))
                image = draw_boxes(image, boxes, config['model']['labels'])

                if show_debug_frame:
                    cv2.imshow("{} - detections".format(os.path.basename(image_path)), image)
                    cv2.waitKey()

                if save_output_video:
                    video_writer.write(np.uint8(image))

                frame_boxes, frame_labels, frame_scores = [], [], []
                for box in boxes:
                    xmin = int(box.xmin * frame_w)
                    ymin = int(box.ymin * frame_h)
                    xmax = int(box.xmax * frame_w)
                    ymax = int(box.ymax * frame_h)
                    frame_boxes.append([xmin, ymin, xmax, ymax])
                    frame_labels.append(str(config["model"]["labels"][box.get_label()]))
                    frame_scores.append(float(box.get_score()))
                detection_dict[frame_i] = {"boxes": frame_boxes, "labels": frame_labels, "scores": frame_scores}

            if write_raw_results_json:
                out_json_file = image_path.replace(".avi", ".json")
                with open(out_json_file, "w") as jf:
                    json.dump(detection_dict, jf)

            if show_debug_frame:
                cv2.destroyAllWindows()

            video_reader.release()
            if save_output_video:
                video_writer.release()

            mean_predict_t_s = total_predict_time_s / num_frames_to_run
            print("Mean predict time: {:.5f}s".format(mean_predict_t_s))

        else:
            image = cv2.imread(image_path)
            boxes = yolo.predict(image)
            image = draw_boxes(image, boxes, config['model']['labels'])
            print(len(boxes), 'boxes are found')
            cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)


if __name__ == '__main__':
    _main_()
