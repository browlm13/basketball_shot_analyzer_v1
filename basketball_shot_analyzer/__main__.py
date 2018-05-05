#!/usr/bin/env python

import math
import logging
import os
import json
import sys
import PIL.Image as Image

# external
import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
# from mpl_toolkits import mplot3d            # 3d plotting
from scipy import stats                     # error rvalue/linear regression
from piecewise.regressor import piecewise   # piecewise regression

# my lib
from image_evaluator.src import image_evaluator
from utils.frames import to_frames

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#
# Writing new images and videos
#

# ext = extension
# source: http://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html
def write_mp4_video(ordered_image_paths, ext, output_mp4_filepath):
    """
    :param ordered_image_paths: array of image path strings to combine into mp4 video file
    :param ext: NOT USED
    :param output_mp4_file path: output file name without extension
    """
    # create output video file directory if it does not exist
    if not os.path.exists(os.path.split(output_mp4_filepath)[0]):
        os.makedirs(os.path.split(output_mp4_filepath)[0])

    # Determine the width and height from the first image
    image_path = ordered_image_paths[0]
    frame = cv2.imread(image_path)
    cv2.imshow('video', frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(output_mp4_filepath, fourcc, 20.0, (width, height))

    for image_path in ordered_image_paths:
        frame = cv2.imread(image_path)
        out.write(frame)  # Write out frame to video

    # Release everything if job is finished
    out.release()


def write_frame_for_accuracy_test(output_directory_path, frame, image_np):
    """
    :param output_directory_path: String of path to directory to write numpy array image to with filename "frame_%s.JPEG" where %s is frame passed as parameter. Path string can be relative or absolute.
    :param frame: int, frame number
    :param image_np: numpy array image to write to file
    """
    # if image output directory does not exist, create it
    if not os.path.exists(output_directory_path): os.makedirs(output_directory_path)

    image_file_name = "%d.JPEG" % frame
    output_file = os.path.join(output_directory_path, image_file_name)

    cv2.imwrite(output_file, image_np)  # BGR color

# list of 4 coordanates for box
def draw_box_image_np(image_np, box, color=(0, 255, 0), width=3):
    """
    :param image_np: numpy array image
    :param box: tuple (x1,x2,y1,y2)
    :param color: tuple (R,G,B) color to draw, defualt is (0,255,0)
    :param width: int, width of rectangle line to draw, default is 3
    :return: returns numpy array of image with rectangle drawn
    """
    (left, right, top, bottom) = box
    cv2.rectangle(image_np, (left, top), (right, bottom), color, 3)
    return image_np


def draw_all_boxes_image_np(image_np, image_info):
    """
    :param image_np: numpy array image
    :param image_info: image_info object with objects containing "box" tuples to draw to numpy array image
    :return: returns numpy array image with all rectangles drawn
    """
    for item in image_info['image_items_list']:
        draw_box_image_np(image_np, item['box'])
    return image_np


def get_category_box_score_tuple_list(image_info, category):
    """
    :param image_info: image/frame info object
    :param category: category/class string
    :returns: returns list of tuples containing objects "box" and "score" of all objects with matching "class"
    """
    score_list = []
    box_list = []
    for item in image_info['image_items_list']:
        if item['class'] == category:
            box_list.append(item['box'])
            score_list.append(item['score'])
    return list(zip(score_list, box_list))


def get_high_score_box(image_info, category, must_detect=True):
    """
    :param image_info: image/frame info object
    :param category: category/class string
    :param must_detect: must_detect boolean. Will throw error if value is True and no matching class is found. Will return None otherwise. default is True
    :returns: "box" of object with highest "score" of selected "class"
    """
    category_box_score_tuple_list = get_category_box_score_tuple_list(image_info, category)

    if len(category_box_score_tuple_list) == 0:
        logger.debug("none detected: %s" % category)
        if must_detect:
            sys.exit()
            high_score_index = 0
            high_score_value = 0

            index = 0
            for item in category_box_score_tuple_list:
                if item[0] > high_score_value:
                    high_score_index = index
                    high_score_value = item[0]
                index += 1

            return category_box_score_tuple_list[high_score_index][1]
        else:
            return None

    high_score_index = 0
    high_score_value = 0

    index = 0
    for item in category_box_score_tuple_list:
        if item[0] > high_score_value:
            high_score_index = index
            high_score_value = item[0]
        index += 1

    return category_box_score_tuple_list[high_score_index][1]


def get_person_mark(person_box):
    """
    :param person_box: tupple (x1,x2,y1,y2)
    :returns: Point (x,y) where x is half the person box's width, and y is 3/4 the person box's height
    """
    # 3/4 height, 1/2 width
    (left, right, top, bottom) = person_box
    width = int((right - left) / 2)
    x = left + width
    height = int((bottom - top) * float(1.0 / 4.0))
    y = top + height
    return (x, y)


def get_ball_mark(ball_box):
    """
    :param ball_box: tupple (x1,x2,y1,y2)
    :returns: Point (x,y) where x and y are half the ball box's width and height
    """
    # 1/2 height, 1/2 width
    (left, right, top, bottom) = ball_box
    width = int((right - left) / 2)
    x = left + width
    height = int((bottom - top) / 2)
    y = top + height
    return (x, y)


def get_angle_between_points(mark1, mark2):
    """
    :param mark1: Point/(x,y)
    :param mark2: Point?(x,y)
    :returns: Angle between points
    """
    x1, y1 = mark1
    x2, y2 = mark2
    radians = math.atan2(y1 - y2, x1 - x2)
    return radians


def get_ball_radius(ball_box, integer=True):
    """
    :param ball_box: tupple (x1,x2,y1,y2)
    :param integer: boolean, True by defualt
    :returns: average between half the box's width and half the box's height, defualt returns int
    """
    (left, right, top, bottom) = ball_box
    xwidth = (right - left) / 2
    ywidth = (bottom - top) / 2
    radius = (xwidth + ywidth) / 2

    if integer: return int(radius)
    return radius


def get_ball_outside_mark(person_box, ball_box):
    """
    :param person_box: tuple (x1,x2,y1,y2)
    :param ball_box: tuple (x1,x2,y1,y2)
    :returns: Return Point located on balls radial surface closest to the person box passed as parameter.
    """
    # mark on circumference of ball pointing towards person mark
    ball_mark = get_ball_mark(ball_box)
    person_mark = get_person_mark(person_box)

    ball_radius = get_ball_radius(ball_box)
    angle = get_angle_between_points(person_mark, ball_mark)

    dy = int(ball_radius * math.sin(angle))
    dx = int(ball_radius * math.cos(angle))

    outside_mark = (ball_mark[0] + dx, ball_mark[1] + dy)
    return outside_mark


# (left, right, top, bottom) = box
def box_area(box):
    """
    :param box: tuple (x1,x2,y1,y2)
    :returns: area of box
    """
    (left, right, top, bottom) = box
    return (right - left) * (bottom - top)


def height_squared(box):
    """
    :param box: tuple (x1,x2,y1,y2)
    :returns: height of box squared
    """
    (left, right, top, bottom) = box
    return (bottom - top) ** 2


# center (x,y), color (r,g,b)
def draw_circle(image_np, center, radius=2, color=(0, 0, 255), thickness=10, lineType=8, shift=0):
    """
    :param image_np: numpy array image
    :param center: tuple (x,y) center of circle to draw
    :param radius: int, radius of circle to draw
    :param color: tuple (R,G,B) color to draw, default is (0,0,255)
    :param thickness: int, thickness of line to draw, default is 10
    :param lineType: opencv parameter
    :param shift: opencv parameter
    :return: returns numpy array of image with circle drawn
    """
    cv2.circle(image_np, center, radius, color, thickness=thickness, lineType=lineType, shift=shift)
    return image_np


def draw_person_ball_connector(image_np, person_mark, ball_mark, color=(255, 0, 0)):
    """
    :param image_np: image to draw line onto
    :param person_mark: tuple (x,y) of one line endpoint
    :param ball_mark: tuple (x,y) of one line endpoint
    :param color: tuple (R,G,B) color of line, default is (255,0,0)
    :return: numpy array image with line drawn
    """
    lineThickness = 7
    cv2.line(image_np, person_mark, ball_mark, color, lineThickness)
    return image_np


def iou(box1, box2):
    """
    :param box1: tuple (x1,x2,y1,y2)
    :param box2: tuple (x1,x2,y1,y2)
    :return: Intersection over union" of two bounding boxes as float (0,1)
    """
    # return "intersection over union" of two bounding boxes as float (0,1)
    paired_boxes = tuple(zip(box1, box2))

    # find intersecting box
    intersecting_box = (max(paired_boxes[0]), min(paired_boxes[1]), max(paired_boxes[2]), min(paired_boxes[3]))

    # adjust for min functions
    if (intersecting_box[1] < intersecting_box[0]) or (intersecting_box[3] < intersecting_box[2]):
        return 0.0

    # compute the intersection over union
    return box_area(intersecting_box) / float(box_area(box1) + box_area(box2) - box_area(intersecting_box))


def load_image_np(image_path):
    """
    :param image_path: string of path to image, relative or non relative
    :returns: returns numpy array of image
    """
    # non relative path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image = Image.open(os.path.join(script_dir, image_path))
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    return image_np


def filter_minimum_score_threshold(image_info_bundle, min_score_thresh):
    """
    :param image_info_bundle: image_info_bundle object
    :param min_score_thresh: Minimum score threshold of objects not to filter out of returned image_info_bundle
    :returns: return filtered image_info_bundle object
    """
    filtered_image_info_bundle = {}
    for image_path, image_info in image_info_bundle.items():
        filtered_image_info_bundle[image_path] = image_info
        filtered_image_items_list = []
        for item in image_info['image_items_list']:
            if item['score'] > min_score_thresh:
                filtered_image_items_list.append(item)
        filtered_image_info_bundle[image_path]['image_items_list'] = filtered_image_items_list
    return filtered_image_info_bundle


def filter_selected_categories(image_info_bundle, selected_categories_list):
    """
    :param image_info_bundle: image_info_bundle object
    :param selected_categories_list: list of strings of categories/classes to keep in returned image_info_bundle object
    :returns: return filtered image_info_bundle object
    """
    filtered_image_info_bundle = {}
    for image_path, image_info in image_info_bundle.items():
        filtered_image_info_bundle[image_path] = image_info
        filtered_image_items_list = []
        for item in image_info['image_items_list']:
            if item['class'] in selected_categories_list:
                filtered_image_items_list.append(item)
        filtered_image_info_bundle[image_path]['image_items_list'] = filtered_image_items_list
    return filtered_image_info_bundle


# saving image_evaluator evaluations
def save_image_directory_evaluations(image_directory_dirpath, image_info_bundle_filepath,
                                     model_list, bool_rule):
    """
    :param image_directory_dirpath: String of directory path to images to be evaluated using image_evaluator
    :param image_info_bundle_filepath: String of file path to output image_info_bundle file, creates if does not exist
    :param model_list: list of models to use in evaluation format: {'name' : 'model name', 'use_display_name' : Boolean, 'paths' : {'frozen graph': "path/to/frozen/interfernce/graph", 'labels' : "path/to/labels/file"}}
    :param bool_rule: boolean statement string for evaluations using class names, normal boolean logic python syntax and any() and num() methods
    """

    # create image evaluator and load models
    ie = image_evaluator.Image_Evaluator()
    ie.load_models(model_list)

    # get path to each frame in video frames directory
    image_path_list = glob.glob(image_directory_dirpath + "/*")

    # evaluate images in directory and write image_boolean_bundle and image_info_bundle to files for quick access
    image_boolean_bundle, image_info_bundle = ie.boolean_image_evaluation(image_path_list, bool_rule)

    # create image evaluator output directory if it does not exist
    if not os.path.exists(os.path.split(image_info_bundle_filepath)[0]):
        os.makedirs(os.path.split(image_info_bundle_filepath)[0])

    with open(image_info_bundle_filepath, 'w+') as file:
        file.write(json.dumps(image_info_bundle))


# loading saved evaluations
def load_image_info_bundle(image_info_bundle_filepath):
    """
    :param image_info_bundle_filepath: String to image_info_bundle_file json file
    :returns: image_info_bundle python dictornary
    """
    with open(image_info_bundle_filepath) as json_data:
        d = json.load(json_data)
    return d


def get_frame_path_dict(video_frames_dirpath):
    """
    :param video_frames_dirpath: String path to directory full of frame images with name format "frame_i.JPEG"
    :return: python dictionary of frame_path_dict format {frame_number:"path/to/image, ...}
    """

    # get path to each frame in video frames directory
    image_path_list = glob.glob(video_frames_dirpath + "/*")

    frame_path_dict = []
    for path in image_path_list:
        # get filename
        filename = os.path.basename(path)

        # strip extension
        filename_wout_ext = filename.split('.')[0]

        # frame_number
        frame = int(filename_wout_ext)

        frame_path_dict.append((frame, path))

    return dict(frame_path_dict)


# return minimum and maximum frame number for frame path dict as well as continuous boolean value
def min_max_frames(frame_path_dict):
    """
    :param frame_path_dict: Python dictionary format {frame_number:"path/to/image, ...}
    :return: minimum frame, maximum frame, and a boolean of whether or not a continuous range exists
    """
    frames, paths = list(frame_path_dict.keys()), list(frame_path_dict.values())

    min_frame, max_frame = min(frames), max(frames)
    continuous = set(range(min_frame, max_frame + 1)) == set(frames)

    return min_frame, max_frame, continuous


def frame_directory_to_video(input_frames_directory, output_video_file):
    """
    :param input_frames_directory: String of path to directory of images 'frame_i.JPEG" format currently supported. Frame range must be continous. Path can be absolute or relative.
    :param output_video_file: String of path of output directory
    """
    # write video
    output_frame_paths_dict = get_frame_path_dict(input_frames_directory)
    min_frame, max_frame, continuous = min_max_frames(output_frame_paths_dict)

    if continuous:
        ordered_frame_paths = []
        for frame in range(min_frame, max_frame + 1):
            ordered_frame_paths.append(output_frame_paths_dict[frame])
        write_mp4_video(ordered_frame_paths, 'JPEG', output_video_file)
    else:
        logger.error("Video Frames Directory %s Not continuous")


#
# pure boundary box image (high score person and ball in image info)
#
def pure_boundary_box_frame(frame_image, image_info):
    # load a frame for size and create black image
    rgb_blank_image = np.zeros(frame_image.shape)

    # get person and ball boxes
    person_box = get_high_score_box(image_info, 'person', must_detect=False)
    ball_box = get_high_score_box(image_info, 'basketball', must_detect=False)

    # draw boxes (filled)
    if person_box is not None:
        (left, right, top, bottom) = person_box
        cv2.rectangle(rgb_blank_image, (left, top), (right, bottom), color=(255, 50, 50), thickness=-1, lineType=8)

    if ball_box is not None:
        (left, right, top, bottom) = ball_box
        cv2.rectangle(rgb_blank_image, (left, top), (right, bottom), color=(30, 144, 255), thickness=-1, lineType=8)

    return rgb_blank_image


#
# stabilize to person mark, scale to ball box (high score person and ball in image info)
#
def stabilize_to_person_mark_frame(frame_image, image_info, image_info_bundle, current_frame):

    # colors
    color_1 = (0, 0, 255)
    color_2 = (0, 255, 0)

    ball_color = color_1

    # load a frame for size and create black image
    # rgb_blank_image = np.zeros(frame_image.shape)
    rgb_blank_image = frame_image

    # find list frames that are part of shots
    shot_frame_ranges = find_shot_frame_ranges(image_info_bundle, single_data_point_shots=False)
    shot_frames = []
    for sfr in shot_frame_ranges:
        shot_frames += list(range(*sfr))
        shot_frames[-1] += 1

    # choose ball color
    if current_frame in shot_frames:
        ball_color = color_2

    # get person and ball boxes
    person_box = get_high_score_box(image_info, 'person', must_detect=False)
    ball_box = get_high_score_box(image_info, 'basketball', must_detect=False)

    if person_box is not None:
        # use person mark as center coordinates
        px, py = get_person_mark(person_box)

        height, width, depth = rgb_blank_image.shape
        center = (int(width / 2), int(height / 2))

        # draw person box
        person_left, person_right, person_top, person_bottom = person_box
        person_width = person_right - person_left
        person_height = person_bottom - person_top

        new_person_left = center[0] - int(person_width / 2)
        new_person_right = center[0] + int(person_width / 2)
        new_person_top = center[1] - int(person_height * (1 / 4))
        new_person_bottom = center[1] + int(person_height * (3 / 4))

        new_person_box = (new_person_left, new_person_right, new_person_top, new_person_bottom)

        if ball_box is not None:

            # use person mark as center coordinates
            bx, by = get_ball_mark(ball_box)

            height, width, depth = rgb_blank_image.shape
            center = (int(width / 2), int(height / 2))

            new_bx = bx - px + center[0]
            new_by = by - py + center[1]
            new_ball_mark = (new_bx, new_by)

            ball_radius = get_ball_radius(ball_box)

            # draw_circle(rgb_blank_image, new_ball_mark)
            # draw_circle(rgb_blank_image, new_ball_mark, radius=ball_radius)

            # old  drawing
            # draw_box_image_np(rgb_blank_image, person_box)
            # draw_circle(rgb_blank_image, (px, py))
            # draw_box_image_np(rgb_blank_image, ball_box) #ball box
            # draw_circle(rgb_blank_image, (bx, by)) #ball circle
            # draw_person_ball_connector(rgb_blank_image, (px,py), (bx,by)) #draw connectors

            # iou overlap
            if iou(person_box, ball_box) > 0:

                #
                # old coordinate drawings
                #

                # ball
                draw_circle(rgb_blank_image, (bx, by), color=ball_color)  # mark
                draw_circle(rgb_blank_image, (bx, by), radius=ball_radius, color=ball_color, thickness=5)  # draw ball
                draw_person_ball_connector(rgb_blank_image, (px, py), (bx, by), color=ball_color)  # connector

                # person
                draw_circle(rgb_blank_image, (px, py), color=ball_color)
            # draw_box_image_np(rgb_blank_image, person_box, color=(0,0,255))

            #
            # new coordinate drawings
            #

            # ball
            # draw_circle(rgb_blank_image, new_ball_mark, color=(0,255,0))   #mark
            # draw_circle(rgb_blank_image, new_ball_mark, radius=ball_radius, color=(0,255,0), thickness=5) #draw ball
            # draw_person_ball_connector(rgb_blank_image, center, new_ball_mark, color=(0,255,0)) # connector

            # person
            # draw_circle(rgb_blank_image, center, color=(0,255,0))
            # draw_box_image_np(rgb_blank_image, new_person_box, color=(0,255,0))

            else:

                #
                # old coordinate drawings
                #

                # ball
                draw_circle(rgb_blank_image, (bx, by), color=ball_color)  # mark
                draw_circle(rgb_blank_image, (bx, by), radius=ball_radius, color=ball_color, thickness=5)  # draw ball
            # draw_person_ball_connector(rgb_blank_image, (px, py), (bx, by), color=(0,255,0)) # connector

            # person
            # draw_circle(rgb_blank_image, (px, py), color=(0,255,0))
            # draw_box_image_np(rgb_blank_image, person_box, color=(0,255,0))

            #
            # new coordinate drawings
            #

            # ball
            # draw_circle(rgb_blank_image, new_ball_mark, color=(0,0,255))   #mark
            # draw_circle(rgb_blank_image, new_ball_mark, radius=ball_radius, color=(0,0,255)) #ball
            # draw_person_ball_connector(rgb_blank_image, center, new_ball_mark, color=(0,0,255)) #connector

            # person
            # draw_circle(rgb_blank_image, center, color=(0,0,255))
            # draw_box_image_np(rgb_blank_image, new_person_box, color=(0,0,255))

    return rgb_blank_image


# run frame cycle and execute function at each step passing current frame path to function, and possibly more
# cycle function should return image after each run
# output frame_path_dict should be equivalent except to output directory
def frame_cycle(image_info_bundle, input_frame_path_dict, output_frames_directory, output_video_file, cycle_function,
                apply_history=False):
    # get minimum and maximum frame indexes
    min_frame, max_frame, continuous = min_max_frames(input_frame_path_dict)

    # frame cycle
    if continuous:

        for frame in range(min_frame, max_frame + 1):
            frame_path = input_frame_path_dict[frame]
            image_info = image_info_bundle[frame_path]

            frame_image = cv2.imread(frame_path)  # read image
            image_np = cycle_function(frame_image, image_info, image_info_bundle, frame)

            if apply_history:

                # TODO: fix weights
                for i in range(frame, min_frame, -1):
                    alpha = 0.1
                    beta = 0.1
                    gamma = 0.5
                    i_frame_path = frame_path_dict[i]
                    i_image_info = image_info_bundle[i_frame_path]
                    i_frame_image = cv2.imread(i_frame_path)  # read image
                    next_image_np = cycle_function(i_frame_image, i_image_info)
                    image_np = cv2.addWeighted(image_np, alpha, next_image_np, beta, gamma)

            # write images
            write_frame_for_accuracy_test(output_frames_directory, frame, image_np)

        # write video
        logger.info("\n\tCreating temporary edited frames directory...\n\tOutput Frames Directory: %s"
                    % output_frames_directory)
        frame_directory_to_video(output_frames_directory, output_video_file)
    else:
        logger.error("not continuous")


# source:
# https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
def group_consecutives(vals, step=1):
    """
    :param vals: list of integers
    :param step: step size, default 1
    :return: list of consecutive lists of numbers from vals (number list).
    """
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


def group_consecutives_by_column(matrix, column, step=1):
    vals = matrix[:, column]
    runs = group_consecutives(vals, step)

    run_range_indices = []
    for run in runs:
        start = np.argwhere(matrix[:, column] == run[0])[0, 0]
        stop = np.argwhere(matrix[:, column] == run[-1])[0, 0] + 1
        run_range_indices.append([start, stop])

    # split matrix into segments (smaller matrices)
    split_matrices = []
    for run_range in run_range_indices:
        start, stop = run_range
        trajectory_matrix = matrix[start:stop, :]
        split_matrices.append(trajectory_matrix)

    return split_matrices


# frame_info_bundle to frame_path_dict
def frame_info_bundle_to_frame_path_dict(input_frame_info_bundle):
    """
    :param input_frame_info_bundle: frame_info_bundle object. Must be only one directory containing all frames
    :return: frame_path_dict python dictionary format {frame_number : "path/to/frame", ...}
    """
    # find unique input_frame_directories, assert there is only 1
    unique_input_frame_directories = list(
        set([os.path.split(frame_path)[0] for frame_path, frame_info in input_frame_info_bundle.items()]))
    assert len(unique_input_frame_directories) == 1

    # return frame_path_dict
    return get_frame_path_dict(unique_input_frame_directories[0])


#
#                           ball collected data points matrix (ball_cdpm)
#
# format 2:
# columns: frame, x1, x2, y1, y2, ball state / iou bool
#
# enum keys: 
#           'frame column', 'x1 column', 'x2' column, 'y1 column', 'y2 column', 'ball state column'     #columns 
#           'no data', 'free ball', 'held ball'                                                         #ball states
#
# "essentially an alternative representation of image_info_bundle"
# to access: frames, xs, ys, state = ball_cdpm.T

def create_ball_cdpm(ball_cdpm_enum, frame_info_bundle):
    # get frame path dict
    frame_path_dict = frame_info_bundle_to_frame_path_dict(frame_info_bundle)

    # get minimum and maximum frame indexes
    min_frame, max_frame, continuous = min_max_frames(input_frame_path_dict)

    #   Matrix - fill with no data
    num_rows = (max_frame + 1) - min_frame
    num_cols = 6  # frame, x1, x2, y1, y2, state (iou bool)
    ball_cdpm = np.full((num_rows, num_cols), ball_cdpm_enum['ball_states']['no_data'])

    # iou boolean lambda function for 'ball mark x column'
    get_ball_state = lambda person_box, ball_box: ball_cdpm_enum['ball_states']['held_ball'] if (
            iou(person_box, ball_box) > 0) else ball_cdpm_enum['ball_states']['free_ball']

    #                   Fill ball collected data points matrix (ball_cdpm)
    #
    #                   'frame', 'x1', 'x2', 'y1', 'y2', 'state'

    index = 0
    for frame in range(min_frame, max_frame + 1):
        frame_path = frame_path_dict[frame]
        frame_info = frame_info_bundle[frame_path]

        # get frame ball box and frame person box
        frame_ball_box = get_high_score_box(frame_info, 'basketball', must_detect=False)
        frame_person_box = get_high_score_box(frame_info, 'person', must_detect=False)

        # frame number column ['column']['frame']
        ball_cdpm[index, ball_cdpm_enum['cdpm_columns']['frame']] = frame

        # ball box columns ['column'][i] for i in 'x1', 'x2', 'y1', 'y2'
        if (frame_ball_box is not None):
            ball_cdpm[index, ball_cdpm_enum['cdpm_columns']['x1']] = frame_ball_box[0]
            ball_cdpm[index, ball_cdpm_enum['cdpm_columns']['x2']] = frame_ball_box[1]
            ball_cdpm[index, ball_cdpm_enum['cdpm_columns']['y1']] = frame_ball_box[2]
            ball_cdpm[index, ball_cdpm_enum['cdpm_columns']['y2']] = frame_ball_box[3]

        # ball state/iou bool column ['column']['state']
        if (frame_ball_box is not None) and (frame_person_box is not None):
            ball_cdpm[index, ball_cdpm_enum['cdpm_columns']['state']] = get_ball_state(frame_person_box, frame_ball_box)

        index += 1

    # return matrix
    return ball_cdpm


def ball_cdpm_boxes_to_marks(ball_cdpm_enum_old, ball_cdpm_enum_new, ball_cdpm):
    """
    :param ball_cdpm_enum_old: python dictionary specifying the column indices of ['cdpm_columns']: 'frame', 'x1', 'x2', 'y1', 'y2', 'state', and the number map for each state ['ball_states']: 'no_data', 'held_ball', 'free_ball'
    :param ball_cdpm_enum_new: python dictionary specifying the column indices of ['cdpm_columns']: 'frame', 'x', 'y', 'state', and the number map for each state ['ball_states']: 'no_data', 'held_ball', 'free_ball'
    :param ball_cdpm: ball_cdpm with columns:  'frame', 'x1', 'x2', 'y1', 'y2', 'state'
    :return: ball_cdpm with columns:  frame, x, y, state - where (x,y) is center point of basketball box or "ball_mark"
    """

    # old cdpm
    frames = ball_cdpm[:, ball_cdpm_enum_old['cdpm_columns']['frame']]
    x1s = ball_cdpm[:, ball_cdpm_enum_old['cdpm_columns']['x1']]
    x2s = ball_cdpm[:, ball_cdpm_enum_old['cdpm_columns']['x2']]
    y1s = ball_cdpm[:, ball_cdpm_enum_old['cdpm_columns']['y1']]
    y2s = ball_cdpm[:, ball_cdpm_enum_old['cdpm_columns']['y2']]
    states = ball_cdpm[:, ball_cdpm_enum_old['cdpm_columns']['state']]

    # new cdpm additions
    num_rows = ball_cdpm.shape[0]
    xs = np.full(num_rows, ball_cdpm_enum_old['ball_states']['no_data'])
    ys = xs.copy()

    # find ball marks and populate new column arrays
    indxs = range(0, num_rows)
    for i, x1, x2, y1, y2 in zip(indxs, x1s, x2s, y1s, y2s):
        ball_box = (x1, x2, y1, y2)
        x, y = get_ball_mark(ball_box)
        xs[i] = x
        ys[i] = y

    # create new cdpm with marks instead of boxes
    new_ball_cdpm = np.array([frames, xs, ys, states]).T

    # arrange columns according to ball_cdpm_enum_new parameter
    new_frames_index = ball_cdpm_enum_new['cdpm_columns']['frame']
    new_xs_index = ball_cdpm_enum_new['cdpm_columns']['x']
    new_ys_index = ball_cdpm_enum_new['cdpm_columns']['y']
    new_states_index = ball_cdpm_enum_new['cdpm_columns']['state']
    column_indices = np.array([new_frames_index, new_xs_index, new_ys_index, new_states_index])

    # re arrange
    new_ball_cdpm = new_ball_cdpm.T
    new_ball_cdpm = new_ball_cdpm[column_indices].T

    # return
    return new_ball_cdpm


#
#                           ball collected data points matrix (ball_cdpm)
#
# format 1:
# columns: frame, x, y, ball state / iou bool
#
# enum keys: 
#           'frame column', 'ball mark x column', 'ball mark y column', 'ball state column'     #columns 
#           'no data', 'free ball', 'held ball'                                                 #ball states
#
# "essentially an alternative representation of image_info_bundle"
# to access: frames, xs, ys, state = ball_cdpm.T
#

def get_ball_cdpm(ball_cdpm_enum, input_frame_path_dict, image_info_bundle):
    # get minimum and maximum frame indexes
    min_frame, max_frame, continuous = min_max_frames(input_frame_path_dict)

    #   Matrix - fill with no data

    num_rows = (max_frame + 1) - min_frame
    num_cols = 4  # frame, ballmark x, ballmark y, ball state (iou bool)
    ball_cdpm = np.full((num_rows, num_cols), ball_cdpm_enum['no data'])

    # iou boolean lambda function for 'ball mark x column'
    get_ball_state = lambda person_box, ball_box: ball_cdpm_enum['held ball'] if (iou(person_box, ball_box) > 0) else \
        ball_cdpm_enum['free ball']

    #                   Fill ball collected data points matrix (ball_cdpm)
    #
    #                   'frame', 'ballmark x', 'ballmark y', 'ball state'

    index = 0
    for frame in range(min_frame, max_frame + 1):
        frame_path = input_frame_path_dict[frame]
        frame_info = image_info_bundle[frame_path]

        # get frame ball box and frame person box
        frame_ball_box = get_high_score_box(frame_info, 'basketball', must_detect=False)
        frame_person_box = get_high_score_box(frame_info, 'person', must_detect=False)

        # frame number column 'frame column'
        ball_cdpm[index, ball_cdpm_enum['frame column']] = frame

        # ball mark column 'ball mark x column', 'ball mark y column' (x,y)
        if (frame_ball_box is not None):
            frame_ball_mark = get_ball_mark(frame_ball_box)
            ball_cdpm[index, ball_cdpm_enum['ball mark x column']] = frame_ball_mark[0]
            ball_cdpm[index, ball_cdpm_enum['ball mark y column']] = frame_ball_mark[1]

        # ball state/iou bool column 'ball state column ''
        if (frame_ball_box is not None) and (frame_person_box is not None):
            ball_cdpm[index, ball_cdpm_enum['ball state column']] = get_ball_state(frame_person_box, frame_ball_box)

        index += 1

    # return matrix
    return ball_cdpm


# get list of shot frame ranges
def find_shot_frame_ranges(frames_info_bundle, std_error_threshold=0.9, single_data_point_shots=False):
    """
    :param frames_info_bundle: frames_info_bundle object to extract shot ranges from
    :param std_error_threshold: Standard error threshold for x axis regression line before performing peicewise regression and returning first line segment as range
    :param single_data_point_shots: Return single data point shot frame ranges boolean, default is set to False
    :return: list of lists [[shot_frame_range_i_start, shot_frame_range_i_stop], ...]
    """
    #
    #   Find shot frame ranges
    #   Mock 2: Assertions: Stable video, 1 person, 1 ball

    #
    #   Build  Format 2 ball data points matrix  (ball_cdpm)
    #                                                   columns: frame, ball box x1, ball box x2, ball box y1, ball box y2, ball state
    #

    ball_cdpm_enum = {
        'ball_states': {
            'no_data': -1,
            'free_ball': 1,
            'held_ball': 0
        },
        'cdpm_columns': {
            'frame': 0,
            'x1': 1,
            'x2': 2,
            'y1': 3,
            'y2': 4,
            'state': 5,
        }
    }

    ball_cdpm = create_ball_cdpm(ball_cdpm_enum, frames_info_bundle)

    #
    # Break ball data points matrix into multiple sub matrices (free_ball_cbdm's) where ball is free
    #               (break around held ball data points)
    #

    # cut ball_cdpm at frames with ball state column value == 'held ball', leaving only free ball datapoints in an array of matrices
    free_ball_cbdm_array = ball_cdpm[
                           ball_cdpm[:, ball_cdpm_enum['cdpm_columns']['state']] != ball_cdpm_enum['ball_states'][
                               'held_ball'], :]  # extract all rows with the ball state column does not equal held ball
    free_ball_cbdm_array = group_consecutives_by_column(free_ball_cbdm_array, ball_cdpm_enum['cdpm_columns'][
        'frame'])  # split into separate matrices for ranges

    #
    # Find shot frame ranges
    #
    shot_frame_ranges = []

    #   for each free_ball_cbdm (sub matrix)
    for i in range(len(free_ball_cbdm_array)):
        possible_trajectory_coordinates = free_ball_cbdm_array[
            i]  # 'possible' ball trajectory coordinates, ranges without held ball states tagged by model

        # extract 'known' ball trajectory coordinates, tagged by model
        # remove missing data points
        known_trajectory_points = possible_trajectory_coordinates[
                                  possible_trajectory_coordinates[:, ball_cdpm_enum['cdpm_columns']['state']] !=
                                  ball_cdpm_enum['ball_states']['no_data'], :]  # extract all rows where there is data
        kframes, kx1s, kx2s, ky1s, ky2s, kstate = known_trajectory_points.T

        kball_boxes = list(zip(kx1s, kx2s, ky1s, ky2s))
        kball_marks = [get_ball_mark(bb) for bb in kball_boxes]

        # enure known ball trajectory has more than 1 data point
        if len(kframes) > 1:

            # find average x and y values
            kxs, kys = zip(*kball_marks)

            #
            #                       peicewise linear regression
            #
            #   Apply if x std error is above threshold
            #   Apply peicewise linear regression to x values.
            #   x values should change linearly if ball is in free flight (ignoring air resistance).
            #   Use peicewise linear regression to find the point at which the free flying ball hits another object/changes its path
            #   Find point of intersection for separate regression lines to find final frame of shot ball trajectory
            #

            # test linear regression std error for x values
            slope, intercept, r_value, p_value, std_err = stats.linregress(kframes, kxs)

            logger.debug("\n\nSTD Error for Regression: %s" % std_err)

            start_frame, stop_frame = kframes[0], kframes[-1]
            shot_frame_range = [start_frame, stop_frame]

            # apply peicewise linear regression only if x std error is above threshold
            # take first line segment start and stop points as shot_frame_range
            if std_err >= std_error_threshold:
                logger.debug("Applying peicewise regression\n")

                # peicewise model
                model = piecewise(kframes, kxs)
                start_frame, stop_frame, coeffs = model.segments[
                    0]  # find start and stopping frame (start_frame, stop_frame) from first line segment, and line formula
                shot_frame_range = [start_frame, stop_frame]

            shot_frame_ranges.append(shot_frame_range)

        # single data point shot frame range - [start/stop frame, start/stop frame]
        if (len(kframes) == 1) and single_data_point_shots:
            shot_frame_range = [kframes[0], kframes[0]]
            shot_frame_ranges.append(shot_frame_range)

    return shot_frame_ranges


def known_boxes_in_frame_range(frame_info_bundle, shot_frame_range, category):
    """
    :param frame_info_bundle: frames_info_bundle object to extract boxes from
    :param shot_frame_range: frame range to extract to extract boxes from
    :param category: String category/class to extract high score boxes from
    :return: [box, frame] where box is a tuple (x1, x2, y1, y2)
    """
    # get frame path dict
    frame_path_dict = frame_info_bundle_to_frame_path_dict(frame_info_bundle)

    # minimum and maximum frames
    min_frame, max_frame = shot_frame_range[0], shot_frame_range[1]

    # Find ball collected boxes
    known_boxes_and_frames = []
    for frame in range(min_frame, max_frame + 1):
        frame_path = frame_path_dict[frame]
        frame_info = frame_info_bundle[frame_path]

        # get frame high scroe ball box
        frame_ball_box = get_high_score_box(frame_info, 'basketball', must_detect=False)

        if (frame_ball_box is not None):
            known_boxes_and_frames.append([frame_ball_box, frame])

    return known_boxes_and_frames


def find_ball_regression_formulas(frame_info_bundle, shot_frame_range, adjust_yvalues=True):
    """
    :param frames_info_bundle: frames_info_bundle object to extract regression formulas from
    :param shot_frame_range: frame range to extract regression formulas from
    :return: regression polynomial coefficients list [pxs,pys]. format pis: [(coeff 0)frame^0, (coeff 1)frame^1, ...] -- (np.polyfit)
    """
    # [TODO] Clean this shit
    # note cannot handle frame ranges of single value

    # get xs, ys, radii's known data points in frame range

    # get known boxes in frame range
    ball_boxes, frames = zip(*known_boxes_in_frame_range(frame_info_bundle, shot_frame_range, 'basketball'))  # tuples

    # get known ball marks
    ball_marks = [get_ball_mark(bb) for bb in ball_boxes]

    # find average x and y values
    xs, ys = zip(*ball_marks)  # tuples

    # find ball radii
    ball_radii = [get_ball_radius(bb, integer=False) for bb in ball_boxes]

    # normalize radii for change only
    # normalize to first radii
    normalized_ball_radii = [r / ball_radii[0] for r in ball_radii]

    # zs_distance_change_coeff - 1/normalized ball radii. this represents the balls distance change from its startposition at the origin
    # greater than 1 is farther away, 2 is twice as far away
    zs_distance_change_coeff = [1 / r for r in normalized_ball_radii]

    # find regression formula then scale to balls distance away
    pzs_change_coeff = np.polyfit(frames, zs_distance_change_coeff, 1)
    total_shot_frames = np.linspace(frames[0], frames[-1], frames[-1] - frames[0])
    # zs_change_coeffs = np.polyval(pzs_change_coeff, total_shot_frames)
    zs_change_coeffs = np.polyval(pzs_change_coeff, frames)

    # ys adjuseted
    # y_adjusted = y/z_change_coeff_matched_range
    #
    ys_adjusted = [y / zcc for y, zcc in zip(ys, zs_change_coeffs)]

    # find x regression polynomial coeffiecnts
    # xs - degreen 1 regression fit
    pxs = np.polyfit(frames, xs, 1)

    # find y regreesion polynomial coeffiecents - currently do not take into account z corrections
    # ys - degreen 2 regression fit
    pys = np.polyfit(frames, ys, 2)

    if adjust_yvalues:
        # find pys with z correction
        pys = np.polyfit(frames, ys_adjusted, 2)

    # return polynomial coefficents
    return [pxs, pys]


def find_normalized_ball_regression_formulas(frame_info_bundle, shot_frame_range, adjust_yvalues=True,
                                             amplify_zslope=True, y_adjust_method="add amplified"):
    """
    :param frames_info_bundle: frames_info_bundle object to extract regression formulas from
    :param shot_frame_range: frame range to extract regression formulas from
    :param adjust_yvalues:
    :param amplify_zslope:
    :return: normalized regression polynomial coefficients to balls radius in pixels list [pxs,pys]. format pis: [(coeff 0)frame^0, (coeff 1)frame^1, ...] -- (np.polyfit)
    """

    # NOTE: cannot handle frame ranges of single value

    #
    # 	Find normalized polynomial coefficients describing shot trajectories for x,y and z axes within given frame range
    #
    #		* use trends in basketball radii to determine z axes function
    #		* adjust y function using z function
    #		* normalize functions to the radius of the basketball in pixels

    #
    #	Retrieve x, y, and radius data points identified by model within the given frame range
    #
    #		* xs, ys, radii, frames, start_frame, stop_frame

    # get boxes, frames, start_frame, and stop_frame in given frame range
    ball_boxes, frames = zip(*known_boxes_in_frame_range(frame_info_bundle, shot_frame_range, 'basketball'))
    start_frame, stop_frame = frames[0], frames[-1]

    # get average x and y values within given frame range (ball_marks)
    ball_marks = [get_ball_mark(bb) for bb in ball_boxes]
    xs, ys = zip(*ball_marks)

    # get ball radii within frame range
    radii = [get_ball_radius(bb, integer=False) for bb in ball_boxes]

    #
    #	Find z axis normalized polynomial coefficients
    #
    #		* normalize basketball radii to start_frames radius
    #		* find regression function describing trends in the changing normalized basketball radius sizes
    #		# amplify normalized radii size regression function to find normalized z function
    #
    #		Justification : If the balls radius is twice as far away it will be half the size

    # normalize basketball radii to start_frames radius
    norm_radii = [r / radii[0] for r in radii]

    # find regression function describing trends in the changing normalized basketball radius sizes
    slope, intercept, r_value, p_value, std_err = stats.linregress(frames, norm_radii)

    # amplify normalized radii size regression function to find normalized z function
    #	* amplify slope of normalized radii size regression function
    #	* damp slope of normalized radii size regression function with r_value
    amplified_slope = (slope) * ((abs(slope) + math.pi) ** 3) * (r_value ** 3)

    # normalized polynomial coefficients for z axis function
    pzs_norm = [slope, intercept]  # final
    pzs_norm_amplified = [amplified_slope, intercept]  # final (if amplify_zslope)

    #
    #	Find y axis normalized polynomial coefficients
    #
    #		* adjust y data points using z function:
    #													formula: yi_adjusted = yi * zi_norm
    #		* normalize adjusted y datapoints to basketball radius in pixels
    #		* find polynomial coefficients for second degree regression fit of normalized and adjusted y values

    # adjust y data points using z function
    zs_norm = np.polyval(pzs_norm, frames)
    zs_norm_amplified = np.polyval(pzs_norm_amplified, frames)  # [CHANGE]: Add

    if adjust_yvalues:

        if y_adjust_method == "norm scale":
            ys = [y * z for y, z in zip(ys, zs_norm)]  # [CHANGE]:

        if y_adjust_method == "amplified scale":  # [CHANGE]:
            ys = [y * z for y, z in zip(ys, zs_norm_amplified)]

        if y_adjust_method == "add amplified":
            ys = [y + z for y, z in zip(ys, zs_norm_amplified)]  # [CHANGE]: Add

    # normalize adjusted y data points to basketball radius in pixels
    ys_norm = [y / r for y, r in zip(ys, radii)]

    # find polynomial coefficients for second degree regression fit of normalized and adjusted y values
    pys_norm = np.polyfit(frames, ys_norm, 2)  # final

    #
    #	Find x axis normalized polynomial coefficients
    #
    #		* normalize adjusted x datapoints to basketball radius in pixels
    #		* find polynomial coefficients for first degree regression fit of normalized and adjusted x values

    # normalize adjusted x datapoints to basketball radius in pixels
    xs_norm = [x / r for x, r in zip(xs, radii)]

    # find polynomial coefficients for first degree regression fit of normalized and adjusted x values
    pxs_norm = np.polyfit(frames, xs_norm, 1)  # final

    #
    # return normalized polynomial coefficents
    #

    if amplify_zslope:
        return [pxs_norm, pys_norm, pzs_norm_amplified]
    return [pxs_norm, pys_norm, pzs_norm]


def pixel_shot_position_vectors(frame_info_bundle, shot_frame_range, extrapolate=False):
    """
    :param frame_info_bundle: frames_info_bundle object to extract pixel position vectors from
    :param shot_frame_range: frame range to extract pixel position vectors from
    :param extrapolate: boolean default False, extrapolate trajectories from known datapoints filling in unkown frames
    :return: return matrix of pixel position vectors with frame, x,y as columns.
    """
    start_frame, stop_frame = shot_frame_range[0], shot_frame_range[1]
    shot_frames = np.linspace(start_frame, stop_frame, stop_frame - start_frame + 1)

    if not extrapolate:
        #
        #	Return raw data identified by model in pixel matrix, mask frames with no data points in given frame range
        #

        ball_cdpm_enum_old = {
            'ball_states': {
                'no_data': -1,
                'free_ball': 1,
                'held_ball': 0
            },
            'cdpm_columns': {
                'frame': 0,
                'x1': 1,
                'x2': 2,
                'y1': 3,
                'y2': 4,
                'state': 5,
            }
        }

        ball_cdpm_enum = {
            'ball_states': {
                'no_data': -1,
                'free_ball': 1,
                'held_ball': 0
            },
            'cdpm_columns': {
                'frame': 0,
                'x': 1,
                'y': 2,
                'state': 3,
            }
        }

        # columns: frame, x1, x2, y1, y2, state
        ball_cdpm = create_ball_cdpm(ball_cdpm_enum_old, frame_info_bundle)

        # convert to : columns: frame, x, y, state
        ball_cdpm = ball_cdpm_boxes_to_marks(ball_cdpm_enum_old, ball_cdpm_enum, ball_cdpm)

        # trim rows to frame range
        frame_column_index = ball_cdpm_enum['cdpm_columns']['frame']
        after_indices = np.where(ball_cdpm[:, frame_column_index] >= start_frame)[0]
        before_indices = np.where(ball_cdpm[:, frame_column_index] <= stop_frame)[0]
        start_index = after_indices[0]
        stop_index = before_indices[-1]
        ball_cdpm = ball_cdpm[start_index:stop_index, :]

        # trim ball state column
        ball_cdpm = np.delete(ball_cdpm, ball_cdpm_enum['cdpm_columns']['state'], 1)

        # trim frame column
        ball_cdpm = np.delete(ball_cdpm, ball_cdpm_enum['cdpm_columns']['frame'], 1)

        # mask rows with no data point
        x_column_index = ball_cdpm_enum['cdpm_columns']['x']
        row_mask_indices = np.where(ball_cdpm[:, x_column_index] == ball_cdpm_enum['ball_states']['no_data'])[0]

        h, w = ball_cdpm.shape
        mask_np = np.zeros((h, w), dtype=int)

        for r in range(h + 1):
            if r in row_mask_indices:
                mask_np[r] = 1

        # create masked pixel matrix
        masked_pixel_matrix = np.ma.array(ball_cdpm, mask=mask_np)

        # return
        return masked_pixel_matrix

    else:
        #
        # return extrapolated pixel matrix
        #		* y values not adjusted
        #		* no mask/empty datapoints in frame range

        pxs, pys = find_ball_regression_formulas(frame_info_bundle, shot_frame_range, adjust_yvalues=False)
        xs = np.polyval(pxs, shot_frames)
        ys = np.polyval(pys, shot_frames)
        pixel_matrix = np.array([xs, ys]).T
        return pixel_matrix


def world_shot_position_vectors(frame_info_bundle, shot_frame_range):
    """
    :param frame_info_bundle: frames_info_bundle object to extract world position vectors from
    :param shot_frame_range: frame range to extract world position vectors from
    :return: return matrix of world position vectors with x,y,z components as columns
    """
    start_frame, stop_frame = shot_frame_range[0], shot_frame_range[1]
    shot_frames = np.linspace(start_frame, stop_frame, stop_frame - start_frame + 1)

    # retrieve normalized polynomial coefficients for x, y and z components within given frame range
    pxs_norm, pys_norm, pzs_norm = find_normalized_ball_regression_formulas(frame_info_bundle,
                                                                            shot_frame_range)  # adjusted ys

    # find x,y and z data points from normalized polynomial coefficients and given frame range
    xs_norm = np.array(np.polyval(pxs_norm, shot_frames))
    ys_norm = np.array(np.polyval(pys_norm, shot_frames))
    zs_norm = np.array(np.polyval(pzs_norm, shot_frames))

    # invert y and z values
    neg = lambda t: t * (-1)
    invert_array = np.vectorize(neg)
    ys_norm = invert_array(ys_norm)
    zs_norm = invert_array(zs_norm)

    # scale to balls actual radius in meters
    ball_radius_meters = 0.12
    xs_meters = np.multiply(xs_norm, ball_radius_meters)
    ys_meters = np.multiply(ys_norm, ball_radius_meters)
    zs_meters = np.multiply(zs_norm, ball_radius_meters)

    # set starting point as origin on all axes
    xs_meters = np.add(xs_meters, -xs_meters[0])
    ys_meters = np.add(ys_meters, -ys_meters[0])
    zs_meters = np.add(zs_meters, -zs_meters[0])

    # return matrix
    return np.array([xs_meters, ys_meters, zs_meters]).T


def get_world_shot_xyzs(frame_info_bundle, shot_frame_range):
    """
    :param frame_info_bundle: frames_info_bundle object to extract world position vectors from
    :param shot_frame_range: frame range to extract world position vectors from
    :return: return list of numpy arrays of world positions [xs,ys,zs]
    """
    world_position_vectors = world_shot_position_vectors(frame_info_bundle, shot_frame_range)
    ball_radius_meters = 1 #0.12  - this is applying it a second time
    world_xs_meters = np.multiply(world_position_vectors[:, 0], ball_radius_meters)
    world_ys_meters = np.multiply(world_position_vectors[:, 1], ball_radius_meters)
    world_zs_meters = np.multiply(world_position_vectors[:, 2], ball_radius_meters)
    return [world_xs_meters, world_ys_meters, world_zs_meters]


# source:https://newtonexcelbach.com/2014/03/01/the-angle-between-two-vectors-python-version/
def py_ang(v1, v2, radians=True):
    """ Returns the angle in radians (by default) between vectors 'v1' and 'v2'  """
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    angle_radians = np.arctan2(sinang, cosang)
    if radians:
        return angle_radians
    return math.degrees(angle_radians)


def get_initial_velocity(frame_info_bundle, shot_frame_range):
    """
    :param frame_info_bundle: frames_info_bundle object to extract initial velocity from
    :param shot_frame_range: frame range to extract initial velocity from
    :return: return initial velocity (m/s)
    """
    FPS = 24  # speed of video
    world_position_vectors = world_shot_position_vectors(frame_info_bundle, shot_frame_range)
    initial_position_vector = world_position_vectors[1]
    initial_velocity_vector = np.multiply(initial_position_vector, FPS)
    initial_velocity = np.linalg.norm(initial_velocity_vector)
    return initial_velocity


def get_launch_angle(frame_info_bundle, shot_frame_range, radians=True):
    """
    :param frame_info_bundle: frames_info_bundle object to extract launch angle from
    :param shot_frame_range: frame range to extract launch angle from
    :param radians: boolean default True, False will return degrees
    :return: return launch angle default radians
    """
    world_position_vectors = world_shot_position_vectors(frame_info_bundle, shot_frame_range)
    initial_position_vector = world_position_vectors[1]
    initial_x_component_vector = np.array([initial_position_vector[0], 0, 0])
    launch_angle = py_ang(initial_x_component_vector, initial_position_vector, radians)
    return launch_angle


def shot_pixel_matrix_to_shot_world_matrix(P, shot_frame_range, frame_info_bundle):
    # Pixel Matrix in shot frame range (P) : cols: x_pixel, y_pixel, z_pixel (missing data points are possible masked)
    # goal: convert raw data points to world coordinates

    #
    #	Retrieve x, y, and radius data points identified by model within the given frame range
    #
    #		* xs, ys, radii, frames, start_frame, stop_frame

    # get boxes, frames, start_frame, and stop_frame in given frame range
    ball_boxes, frames = zip(*known_boxes_in_frame_range(frame_info_bundle, shot_frame_range, 'basketball'))
    start_frame, stop_frame = frames[0], frames[-1]

    # get average x and y values within given frame range (ball_marks)
    ball_marks = [get_ball_mark(bb) for bb in ball_boxes]
    xs, ys = zip(*ball_marks)

    # get ball radii within frame range
    radii = [get_ball_radius(bb, integer=False) for bb in ball_boxes]
    norm_radii = [r / radii[0] for r in radii]

    # shot frame range
    start_frame, stop_frame = shot_frame_range[0], shot_frame_range[1]
    shot_frames = np.linspace(start_frame, stop_frame, stop_frame - start_frame)  # +1)

    # world scale
    bb_radius_world = .12  # determines units of world

    # find regression function describing trends in the changing normalized basketball radius sizes
    slope, intercept, r_value, p_value, std_err = stats.linregress(frames, norm_radii)

    # amplify normalized radii size regression function to find normalized z function
    #	* amplify slope of normalized radii size regression function
    #	* damp slope of normalized radii size regression function with r_value
    amplified_slope = (slope) * ((abs(slope) + math.pi) ** 3) * (r_value ** 3)

    # normalized polynomial coefficients for z axis function
    pradii_amplified = [amplified_slope, intercept]
    pradii = [slope, intercept]
    radii_hat = lambda f: np.polyval(pradii, f)
    radii_hats = map(radii_hat, shot_frames)

    pixel_xs, pixel_ys = P.T

    # print(bb_radius_world / radii_hat(start_frame))

    camera_scaler = bb_radius_world / radii_hat(start_frame)

    # camera_x(f) = pixel_x(f) * camera_scaler
    # camera_y(f) = pixel_y(f) * camera_scaler
    # camera_z(f) = pixel_z(f) * camera_scaler = 0

    # world_x(f) = camera_x(f) = pixel_x(f) * camera_scaler
    # world_y(f) = camera_y(f)/world_z(f) = pixel_y(f) * camera_scaler / world_z(f) = pixel_y(f) / radii_hat(f)
    # world_z(f) = radii_hat(f) * camera_scaler

    camera_xs = np.multiply(pixel_xs, camera_scaler)
    camera_ys = np.multiply(pixel_ys, camera_scaler)
    camera_zs = np.full(camera_xs.shape, 0)

    world_xs = camera_xs
    world_zs = np.multiply(radii_hat(shot_frames), camera_scaler)
    world_ys = np.divide(camera_ys, world_zs)

    # array mask
    array_mask = np.ma.getmask(world_xs)
    # mask zs
    world_zs = np.ma.masked_array(world_zs, mask=array_mask)

    #
    # translate
    #
    world_xs = np.subtract(world_xs, world_xs[0])
    world_ys = np.subtract(world_ys, world_ys[0])
    world_zs = np.subtract(world_zs, world_zs[0])

    # invert ys and zs
    neg = lambda t: t * (-1)
    invert_array = np.vectorize(neg)
    world_ys = invert_array(world_ys)
    world_zs = invert_array(world_zs)

    world_xs = np.ma.masked_array(world_xs, mask=array_mask)
    world_ys = np.ma.masked_array(world_ys, mask=array_mask)
    world_zs = np.ma.masked_array(world_zs, mask=array_mask)

    # return matrix
    return np.array([world_xs, world_ys, world_zs]).T


# get error of least squares fit
def get_error(xs, xs_hat):
    assert len(xs) == len(xs_hat)
    squared_error = 0
    for i in range(len(xs)):
        squared_error += abs(xs[i] - xs_hat[i])
    return (math.sqrt(squared_error))


# error of slope fit (degree 2): m=slope, p2, xs, ys
def error_of_slope_fit(m, p2_old, xs, ys):
    p2 = copy.deepcopy(p2_old)
    p2[0] = m
    y_corrections = []

    # p2 is formula for correction
    c0 = np.polyval(p2, xs[0])
    for x in xs:
        ci = np.polyval(p2, x)
        correction = ci - c0
        y_corrections.append(correction)
    y_corrections = np.array(y_corrections)
    corrected_ys = np.add(y_corrections, ys)

    p2_cy = np.polyfit(xs, corrected_ys, 2)
    p2_corrected_ys = np.polyval(p2_cy, xs)

    return get_error(p2_corrected_ys, corrected_ys)


#
#
#                                           Main
#
#

if __name__ == '__main__':

    #
    # Retrieve mp4 input video file path and output directory path from script arguments
    #

    if len(sys.argv) != 3:
        logger.error("Correct usage requires .mp4 file path and an output directory path as script parameters. Program Terminating.")
    assert len(sys.argv) == 3

    # Retrieve mp4 video file path from script arguments
    mp4_filepath = sys.argv[1]
    mp4_filename = os.path.split(mp4_filepath)[1]

    # Final output directory path for edited video and shot plots .png's
    output_directory_path = sys.argv[2]
    output_filepath_template = output_directory_path
    if output_filepath_template[-1] is not '/':
        output_filepath_template += '/'
    output_filepath_template += "%s"

    # log parameters
    logger.info("\n\tInput video file path: \"%s\"...\n\tOutput Directory: \"%s\"..." % (mp4_filepath, output_directory_path))

    #
    # Split video into temporary directory of frames
    #
    logger.info("\n\tSegmenting video: \"%s\"..." % mp4_filename)
    video_frames_dirpath = os.path.join(os.path.split(os.path.realpath(__file__))[0], "working_data/temporary_files/TMP_original_frames/")
    to_frames(mp4_filepath, video_frames_dirpath, directory_name=None)

    #
    # Initial Model valuation
    #

    # Tensorflow models
    BASKETBALL_MODEL = {'name' : 'basketball_model_v1', 'use_display_name' : False, 'paths' : {'frozen graph': "image_evaluator/models/basketball_model_v1/frozen_inference_graph/frozen_inference_graph.pb", 'labels' : "image_evaluator/models/basketball_model_v1/label_map.pbtxt"}}
    PERSON_MODEL = {'name' : 'ssd_mobilenet_v1_coco_2017_11_17', 'use_display_name' : True, 'paths' : {'frozen graph': "image_evaluator/models/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph/frozen_inference_graph.pb", 'labels' : "image_evaluator/models/ssd_mobilenet_v1_coco_2017_11_17/mscoco_label_map.pbtxt"}}
    BASKETBALL_PERSON_MODEL = {'name' : 'person_basketball_model_v1', 'use_display_name' : False, 'paths' : {'frozen graph': "image_evaluator/models/person_basketball_model_v1/frozen_inference_graph/frozen_inference_graph.pb", 'labels' : "image_evaluator/models/person_basketball_model_v1/label_map.pbtxt"}}

    # message
    logger.info("\n\tAnalyzing video: %s" % mp4_filename)

    # evaluation model collection
    model_collection_name = "model_collection_1"
    logger.info("\n\tModel collection name: %s" % model_collection_name)
    model_collection = [BASKETBALL_MODEL, PERSON_MODEL]
    model_names = []
    for model in model_collection:
        model_names.append(model['name'])
    logger.info("\n\tTensorflow models used: \n\t\t%s" % '\n\t\t'.join(model_names))

    # output images and video directories for checking
    output_frames_directory = os.path.join(os.path.split(os.path.realpath(__file__))[0], "working_data/temporary_files/TMP_edited_frames/")
    output_video_file = output_filepath_template % "machine_vision.mp4"

    # save info_bundle to file "image_evaluator_output/COLLECTION_NAME/VIDEO_FILENAME_image_info_bundle.json"
    image_info_bundle_filepath = os.path.join(os.path.split(os.path.realpath(__file__))[0],
                                              "working_data/cache/image_evaluator_output/%s/%s_image_info_bundle.json" %
                                              (model_collection_name, mp4_filename.split('.')[0]))
    logger.info("\n\tCached Storage Of Model Evaluations At:\n\t\t%s" % image_info_bundle_filepath)

    # bool rule - any basketball or person above an accuracy score of 40.0
    bool_rule = "any('basketball', 40.0) or any('person', 40.0)"

    # check if clip has already been analyzed by model collection
    if not os.path.isfile(image_info_bundle_filepath):
        #
        #     evaluate frame directory
        #               and
        #   save to files for quick access
        #

        logger.info("Analyzing...")
        # perform frame evaluations with image evaluator
        save_image_directory_evaluations(video_frames_dirpath, image_info_bundle_filepath, model_collection, bool_rule)
        logger.info("Cached New Model Collection Evaluations.")
    else:
        logger.info("Model Collection Evaluations In Cache.")

    #
    #   load previously evaluated frames
    #

    # load saved image_info_bundle
    image_info_bundle = load_image_info_bundle(image_info_bundle_filepath)

    #
    # filter model evaluations
    #

    # filter detected objects not in selected categories or bellow minimum score threshold
    selected_categories_list = ['basketball', 'person']
    min_score_thresh = 10.0
    logger.info("Filtering model evaluations with settings:\n\tminimum score threshold: %s,\n\tselected categories: %s" % (min_score_thresh, ' '.join(selected_categories_list)))
    image_info_bundle = filter_selected_categories(filter_minimum_score_threshold(image_info_bundle, min_score_thresh), selected_categories_list)

    # get frame image paths in order
    input_frame_path_dict = get_frame_path_dict(video_frames_dirpath)

    #
    #   Call function for frame cycle
    #

    frame_cycle(image_info_bundle, input_frame_path_dict, output_frames_directory, output_video_file,
                stabilize_to_person_mark_frame)


    #
    #   Intelligently extract all shot frame ranges withing video
    #

    # get shot_frame_ranges
    shot_frame_ranges = find_shot_frame_ranges(image_info_bundle, single_data_point_shots=False)

    #
    #   Extract Extrapolated World Shot Data Points
    #
    
    world_data_matrices = 	[] #[xs_meters, ys_meters, zs_meters, shot_frames]	# get world shot data
    pixel_data_matrices = 	[] #[xs, ys, shot_frames]							# get pixel data

    for sfr in shot_frame_ranges:

        # world data points
        start_frame, stop_frame = sfr[0], sfr[1]
        shot_frames = np.linspace(start_frame, stop_frame, stop_frame-start_frame+1)
        initial_velocity = get_initial_velocity(image_info_bundle, sfr)
        launch_angle_degrees = get_launch_angle(image_info_bundle, sfr, radians=False)
        xs_meters, ys_meters, zs_meters = get_world_shot_xyzs(image_info_bundle, sfr)

        # get shot world matrix (world_shot_position_vectors)
        shot_world_matrix = world_shot_position_vectors(image_info_bundle, sfr)
        world_data_matrices.append(shot_world_matrix)

        #
        # 	P - get pixel matrix 2D
        #	
        #	* matrix of (x,y) vectors as rows
        #

        # pixel_matrix = find_ball_regression_formulas(image_info_bundle, sfr, adjust_values=False)
        shot_pixel_matrix = pixel_shot_position_vectors(image_info_bundle, sfr, extrapolate=True)
        pixel_data_matrices.append(shot_pixel_matrix)

        #
        #	Create Analysis Plot
        #

        try:
            shot_xs_meters, shot_ys_meters, shot_zs_meters, shot_frames, initial_velocity, launch_angle_degrees = \
                xs_meters, ys_meters, zs_meters, shot_frames, initial_velocity, launch_angle_degrees  # world_data[0]

            ax = plt.axes(projection='3d')
            ax.set_aspect('equal')
            scat = ax.scatter(shot_xs_meters, shot_ys_meters, shot_zs_meters, c=(1,.45,0), edgecolors=(1,.3,0))
            ax.plot(shot_xs_meters, shot_ys_meters, shot_zs_meters)
            ax.set_xlabel('Xs meters', linespacing=3.2)
            ax.set_ylabel('\tYs meters', linespacing=3.2)
            ax.set_zlabel('\tZs meters', linespacing=3.2)
            ax.yaxis.set_rotate_label(False)
            ax.zaxis.set_rotate_label(False)
            ax.tick_params(direction='out', length=2, width=1, colors='b', labelsize='small')
            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([shot_xs_meters.max()-shot_xs_meters.min(), shot_ys_meters.max()-shot_ys_meters.min(), shot_zs_meters.max()-shot_zs_meters.min()]).max()
            Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(shot_xs_meters.max()+shot_xs_meters.min())
            Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(shot_ys_meters.max()+shot_ys_meters.min())
            Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(shot_zs_meters.max()+shot_zs_meters.min())
            # Comment or uncomment following both lines to test the fake bounding box:
            for xb, yb, zb in zip(Xb, Yb, Zb):
               ax.plot([xb], [yb], [zb], 'w')

            figure_text = "Video %s\nInitial Velocity %f m/s\nLaunch Angle %f degrees" % (mp4_filename, initial_velocity, launch_angle_degrees)
            plt.figtext(.25, 0.125, figure_text, style='italic',
            bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})
            ax.view_init(elev=140, azim=-90)

            ax.scatter(shot_xs_meters[0], shot_ys_meters[0], shot_zs_meters[0], c='None', s=100,edgecolors='g', linewidths=2)
            plt.grid()

            #
            # display plot
            #
            # plt.show()

            figure_name = "shot_%s.png" % str(shot_frame_ranges.index(sfr) + 1)
            figure_file_path = output_filepath_template % figure_name

            # create graph output figure directory if it does not exist
            if not os.path.exists(os.path.split(figure_file_path)[0]):
                os.makedirs(os.path.split(figure_file_path)[0])

            # write figure
            plt.savefig(figure_file_path)

        except:
            raise

        #
        #   remove temporary directories "tmp_edited_frames", "tmp_original_frames")
        #
        logger.info("Cleaning temporary files...")
        temporary_file_glob_phrase = os.path.join(os.path.split(os.path.realpath(__file__))[0], "working_data/temporary_files/TMP_*/*.*")
        for tmp_file in glob.glob(temporary_file_glob_phrase):
            os.remove(tmp_file)

        # exit program
        logger.info("Program complete.")



