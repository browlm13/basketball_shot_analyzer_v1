# python3

"""
	
	Image Evaluator Class


	img_eval = Image_Evaluator()

	# Loading Models - Todo: store in file so only model name has to be used

	BASKETBALL_MODEL = {'name' : 'basketball_model', 'paths' : {'frozen graph': PATH_TO_FROZEN_GRAPH, 'labels' : PATH_TO_LABELS}}
	PERSON_MODEL = {'name' : 'person_model', 'paths' : {'frozen graph': PATH_TO_FROZEN_GRAPH, 'labels' : PATH_TO_LABELS}}

	img_eval.load_models([BASKETBALL_MODEL, PERSON_MODEL])


	todo: img_eval.annotate_directory(image_directory, annotations_directory) #Add selected categories and minscores

	todo: cropping


"""
import numpy as np
import os
import logging
from PIL import Image
import PIL.Image as Image
import xml.etree.ElementTree as ET
from xml.dom import minidom
import tensorflow as tf
# from utils import label_map_util
from image_evaluator.src.utils import label_map_util

import glob
import shutil

# from shutil import copyfile
# from shutil import copy


# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Image_Evaluator:

    def __init__(self):
        self.models = []
        self.categories = {}

    def load_models(self, model_list):

        # Todo: ensure existence
        self.models = model_list

        # determine categories
        for m in self.models:
            # get each models label dict
            m['categories'] = label_map_util.get_label_map_dict(m['paths']['labels'],
                                                                use_display_name=m['use_display_name'])

        # go through models, for each unique category list all models that can identify, use first as evaluation model
        for m in self.models:
            for key in m['categories']:
                if key in self.categories:
                    self.categories[key]['models'].append(m['name'])
                else:
                    self.categories[key] = {'models': [m['name']], 'evaluation_model': m['name']}

    # set all evaluation models used (what needs to be loaded into memory for image evaluation)
    def get_evaluation_models(self):
        evaluation_models = []
        for c in self.categories:
            if self.categories[c]['evaluation_model'] not in evaluation_models:
                evaluation_models.append(self.categories[c]['evaluation_model'])

        return evaluation_models

    def set_category_evaluation_model(self, category_name, model_name):
        self.categories[category_name]['evaluation_model'] = model_name

    # path, folder, filename
    def get_path_data(self, path):
        folder = os.path.basename(os.path.dirname(path))
        filename = os.path.basename(path)
        return path, folder, filename

    def get_model_path(self, model_name, file_name):
        path = ""
        for model in self.models:
            if model['name'] == model_name:
                path = model['paths'][file_name]
        return path

    def get_model_categories_dict(self, model_name):
        for model in self.models:
            if model['name'] == model_name:
                return model['categories']

    def get_model_evaluation_categories(self, model_name):
        evaluation_categories = []
        for c in self.categories:
            if self.categories[c]['evaluation_model'] == model_name:
                evaluation_categories.append(c)
        return evaluation_categories

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def image_dimensions(self, image_np):
        image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
        return image_pil.size

    #
    # Writing Image XML annotations
    #

    def swap_exentsion(self, full_filename, new_extension):
        template = "%s.%s"  # filename, extension
        filename_base, old_extension = os.path.splitext(full_filename)
        return template % (filename_base, new_extension.strip('.'))

    def generate_new_filename(self, output_directory_path, image_info, new_extension):
        new_filename = self.swap_exentsion(image_info['image_filename'], new_extension)
        full_path = os.path.join(output_directory_path, new_filename)
        return full_path

    def generate_xml_string(self, image_info):

        image_data = {}
        image_data['path'] = image_info['image_path']
        image_data['folder'] = image_info['image_folder']
        image_data['filename'] = image_info['image_filename']
        image_data['width'] = image_info['image_width']
        image_data['height'] = image_info['image_height']
        image_data['depth'] = 3

        # unspecifeid
        image_data['database'] = 'NA'
        image_data['segmented'] = 0

        image_data['objects'] = []
        for item in image_info['image_items_list']:
            o = {}
            o['name'] = item['class']

            xmin, xmax, ymin, ymax = item['box']
            o['xmin'] = xmin
            o['ymin'] = ymin
            o['xmax'] = xmax
            o['ymax'] = ymax

            # unspecifeid
            o['pose'] = 'Unspecified'
            o['truncated'] = 0
            o['difficult'] = 0

            image_data['objects'].append(o)

        # create XML
        annotation_tag = ET.Element('annotation')

        folder_tag = ET.SubElement(annotation_tag, 'folder')
        folder_tag.text = image_data['folder']

        filename_tag = ET.SubElement(annotation_tag, 'filename')
        filename_tag.text = image_data['filename']

        path_tag = ET.SubElement(annotation_tag, 'path')
        path_tag.text = image_data['path']

        source_tag = ET.SubElement(annotation_tag, 'source')
        database_tag = ET.SubElement(source_tag, 'database')
        database_tag.text = image_data['database']

        size_tag = ET.SubElement(annotation_tag, 'size')
        width_tag = ET.SubElement(size_tag, 'width')
        width_tag.text = str(image_data['width'])
        height_tag = ET.SubElement(size_tag, 'height')
        height_tag.text = str(image_data['height'])
        depth_tag = ET.SubElement(size_tag, 'depth')
        depth_tag.text = str(image_data['depth'])

        segmented_tag = ET.SubElement(annotation_tag, 'segmented')
        segmented_tag.text = str(0)

        for o in image_data['objects']:
            object_tag = ET.SubElement(annotation_tag, 'object')
            name_tag = ET.SubElement(object_tag, 'name')
            name_tag.text = o['name']
            pose_tag = ET.SubElement(object_tag, 'pose')
            pose_tag.text = o['pose']
            truncated_tag = ET.SubElement(object_tag, 'truncated')
            truncated_tag.text = str(o['truncated'])
            difficult_tag = ET.SubElement(object_tag, 'difficult')
            difficult_tag.text = str(o['difficult'])
            bndbox_tag = ET.SubElement(object_tag, 'bndbox')
            xmin_tag = ET.SubElement(bndbox_tag, 'xmin')
            xmin_tag.text = str(o['xmin'])
            ymin_tag = ET.SubElement(bndbox_tag, 'ymin')
            ymin_tag.text = str(o['ymin'])
            xmax_tag = ET.SubElement(bndbox_tag, 'xmax')
            xmax_tag.text = str(o['xmax'])
            ymax_tag = ET.SubElement(bndbox_tag, 'ymax')
            ymax_tag.text = str(o['ymax'])

        # return ET.tostring(annotation_tag).decode('utf-8')
        dom = minidom.parseString(ET.tostring(annotation_tag).decode('utf-8'))
        return dom.toprettyxml(indent='\t')

    def write_xml_file(self, image_info, outpath):

        # if directorydoes not exist, create it
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        xml_string = self.generate_xml_string(image_info)
        xml_filename = self.generate_new_filename(outpath, image_info, 'xml')

        with open(xml_filename, "w") as f:
            f.write(xml_string)

    def filter_minimum_score_threshold(self, image_info_bundel, min_score_thresh):
        filtered_image_info_bundel = {}
        for image_path, image_info in image_info_bundel.items():
            filtered_image_info_bundel[image_path] = image_info
            filtered_image_items_list = []
            for item in image_info['image_items_list']:
                if item['score'] > min_score_thresh:
                    filtered_image_items_list.append(item)
            filtered_image_info_bundel[image_path]['image_items_list'] = filtered_image_items_list
        return filtered_image_info_bundel

    def filter_selected_categories(self, image_info_bundel, selected_categories_list):
        filtered_image_info_bundel = {}
        for image_path, image_info in image_info_bundel.items():
            filtered_image_info_bundel[image_path] = image_info
            filtered_image_items_list = []
            for item in image_info['image_items_list']:
                if item['class'] in selected_categories_list:
                    filtered_image_items_list.append(item)
            filtered_image_info_bundel[image_path]['image_items_list'] = filtered_image_items_list
        return filtered_image_info_bundel

    def _image_info(self, category_index, selected_categories, image_np, boxes, scores, classes,
                    min_score_thresh=0.0001):
        # retrieve image size
        image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
        im_width, im_height = image_pil.size

        # box, class, score
        item_list = []

        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > min_score_thresh:
                item = {}

                #
                # box
                #
                normalized_box = tuple(boxes[i].tolist())
                n_ymin, n_xmin, n_ymax, n_xmax = normalized_box
                box = (int(n_xmin * im_width), int(n_xmax * im_width), int(n_ymin * im_height),
                       int(n_ymax * im_height))  # (left, right, top, bottom)
                item['box'] = box

                #
                # class name
                #
                class_name = 'NA'
                if classes[i] in category_index.keys():
                    class_name = str(category_index[classes[i]]['name'])

                item['class'] = class_name

                #
                # detection score
                #
                item['score'] = 100 * scores[i]

                # add if class is in selected_classes, to ensure only evaluation model is evalutating
                if item['class'] in selected_categories:
                    item_list.append(item)

        return item_list

    def get_image_info(self, image_path_list, min_score_thresh=None):

        # TODO: get evaluated fps running average and predict time to completion. Log final time elapsed for evaluation
        # TODO: speed this shit up
        # TODO: spell image_info_bund"el" correctly

        # loading status variables for log
        num_models = len(self.get_evaluation_models())
        num_images = len(image_path_list)
        total_num_evaluations = num_images * num_models
        evaluation_count = 0

        # Initialize image_info_bundel dictionary
        image_info_bundel = dict((image_path,
                                  {'image_items_list': [], 'image_folder': '', 'image_filename': '', 'image_path': '',
                                   'image_height': -1, 'image_width': -1}) for image_path in
                                 image_path_list)  # key= path, value is combined item list

        # for each unique model evaluator in categories list perform detection
        for model_name in self.get_evaluation_models():

            # Load a (frozen) Tensorflow model into memory.
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.get_model_path(model_name, 'frozen graph'), 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

            # Loading label map
            # convert label maps map indices to category names using a dictionary to mapping integers to string labels
            path_to_labels = self.get_model_path(model_name, 'labels')
            label_map = label_map_util.load_labelmap(path_to_labels)
            categories_dict = self.get_model_categories_dict(model_name)
            num_classes = len(categories_dict)
            categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                        use_display_name=True)
            category_index = label_map_util.create_category_index(categories)

            #
            # Detection
            #

            with detection_graph.as_default():
                with tf.Session(graph=detection_graph) as sess:

                    image_tensor = detection_graph.get_tensor_by_name(
                        'image_tensor:0')  # Definite input and output Tensors for detection_graph
                    detection_boxes = detection_graph.get_tensor_by_name(
                        'detection_boxes:0')  # Each box represents a part of the image where a particular object \
                    # was detected.
                    detection_scores = detection_graph.get_tensor_by_name(
                        'detection_scores:0')  # Each score represent how level of confidence for each of the objects.
                    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                    #
                    # Image Detection Loop
                    #

                    for image_path in image_path_list:

                        # log current model and image path used in evaluation
                        logger.info("\n\tEvaluating image: %s,\n\tWith model: %s" % (image_path, model_name))
                        completion_percentage = evaluation_count / total_num_evaluations
                        logger.info("\n\t\tCompletion Percentage: %f" % completion_percentage)

                        #
                        # prepare image for model input
                        #

                        # Non relative path
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        image = Image.open(os.path.join(script_dir, image_path))
                        image_np = self.load_image_into_numpy_array(image)
                        image_np_expanded = np.expand_dims(image_np, axis=0)  # Expand dimensions since the model \
                        # expects images to have shape: [1, None, None, 3]

                        #
                        # Detection
                        #

                        (boxes, scores, classes, num) = sess.run(
                            [detection_boxes, detection_scores, detection_classes, num_detections],
                            feed_dict={image_tensor: image_np_expanded})

                        #
                        # Reformat results
                        #

                        boxes = np.squeeze(boxes)
                        scores = np.squeeze(scores)
                        classes = np.squeeze(classes).astype(np.int32)

                        #
                        # Get selected items (box, class, score)
                        #

                        # selected classes are all categories current model is set to evaluate
                        selected_categories = self.get_model_evaluation_categories(model_name)

                        image_items_list = []
                        if min_score_thresh is not None:
                            mst_decimal = min_score_thresh * 0.01  # convert to decimal
                            image_items_list = self._image_info(category_index, selected_categories, image_np, boxes,
                                                                scores, classes, mst_decimal)
                        else:
                            image_items_list = self._image_info(category_index, selected_categories, image_np, boxes,
                                                                scores, classes)

                        # add to / combine image items list
                        image_info_bundel[image_path]['image_items_list'] += image_items_list

                        #
                        # meta data - PLEASE STORE FOR USE IN XML ANNOTATIONS
                        #

                        image_path, image_folder, image_filename = self.get_path_data(image_path)
                        image_height, image_width = self.image_dimensions(image_np)
                        image_info_bundel[image_path]['image_path'] = image_path
                        image_info_bundel[image_path]['image_folder'] = image_folder
                        image_info_bundel[image_path]['image_filename'] = image_filename
                        image_info_bundel[image_path]['image_height'] = image_height
                        image_info_bundel[image_path]['image_width'] = image_width

                        # increment evaluation count
                        evaluation_count += 1

        return image_info_bundel

    def remove_string_start_end_whitespace(self, string):
        if string[0] == ' ':
            string = string[1:]
        if string[-1] == ' ':
            string = string[:-1]
        return string

    def category_2_symbol(self, category_name):
        return category_name.strip()

    def _any(self, category_name, min_score, image_items_list):
        """ return True if one or more of the category name was detected above minimum score """
        for item in image_items_list:
            if (item['class'] == category_name) and (item['score'] > min_score): return True
        return False

    def _num(self, category_name, min_score, image_items_list):
        """ return number of  the category name detected above minimum score """
        num_detected = 0
        for item in image_items_list:
            if (item['class'] == category_name) and (item['score'] > min_score): num_detected += 1
        return num_detected

    def boolean_image_evaluation(self, image_path_list, boolean_categories_present):
        """ accepts list of paths to images and common boolean expression of categories present ex: any('person',30.0) or (num('basketball', 60.0) > 2)"""

        image_info_bundel = self.get_image_info(image_path_list)
        image_boolean_bundel = dict(
            (image_path, False) for image_path in image_path_list)  # key= path, value is set to false initally

        for image_path, image_info in image_info_bundel.items():
            any = lambda category_name, min_score: self._any(category_name, min_score, image_info['image_items_list'])
            num = lambda category_name, min_score: self._num(category_name, min_score, image_info['image_items_list'])
            scope = locals()

            image_boolean_bundel[image_path] = eval(boolean_categories_present, scope)

        return image_boolean_bundel, image_info_bundel

    def move_images_bool_rule(self, input_image_directory_path, image_output_directory_path, bool_rule,
                              annotations_output_directory_path=False, annotations_min_score_thresh=None,
                              annotations_selected_category_list=None):
        """
        Given input directory of images (currently only supports JPEG), move selected images that satisfy bool rule to
        new directory, create annotation directory (xml) if specified.
        """

        # get all image paths in directory
        accpeted_extensions = ['jpg', 'JPEG', 'jpeg']
        image_path_list = []
        for extension in accpeted_extensions:
            glob_phrase = os.path.join(input_image_directory_path, '*.' + extension)

            for image_path in glob.glob(glob_phrase):

                # check image can be reshaped tmp
                try:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    image = Image.open(os.path.join(script_dir, image_path))
                    image_np = self.load_image_into_numpy_array(image)
                    image_np_expanded = np.expand_dims(image_np, axis=0)  # Expand dimensions since the model

                    # Add image path to image path list
                    image_path_list += [image_path]

                except:
                    logger.error("error loading: %s" % image_path)

        # evaluate
        image_boolean_bundel, image_info_bundel = self.boolean_image_evaluation(image_path_list, bool_rule)

        # if image output directory does not exist, create it
        if not os.path.exists(image_output_directory_path): os.makedirs(image_output_directory_path)

        # copy images over with same basename
        for image_path, copy_bool in image_boolean_bundel.items():
            if copy_bool:
                shutil.copy(image_path, image_output_directory_path)

        # annotations
        # if image output directory does not exist, create it
        if annotations_output_directory_path is not False:
            if not os.path.exists(annotations_output_directory_path): os.makedirs(annotations_output_directory_path)

            # filter selected categories and min score threshold for image_info_bundel
            if annotations_selected_category_list is not None:
                image_info_bundel = self.filter_selected_categories(image_info_bundel,
                                                                    annotations_selected_category_list)
            if annotations_min_score_thresh is not None:
                image_info_bundel = self.filter_minimum_score_threshold(image_info_bundel, annotations_min_score_thresh)

            # change image location data and write xml file
            for image_path, image_info in image_info_bundel.items():

                # if bool statment is true
                if image_boolean_bundel[image_path]:
                    # change image location info
                    new_image_info = image_info

                    new_image_filename = os.path.basename(image_path)  # same technically
                    new_image_folder = os.path.basename(image_output_directory_path)
                    new_image_path = os.path.join(image_output_directory_path, new_image_filename)

                    new_image_info['image_path'] = new_image_path
                    new_image_info['image_folder'] = new_image_folder
                    new_image_info['image_filename'] = new_image_filename

                    # write
                    self.write_xml_file(new_image_info, annotations_output_directory_path)


def run():
    pass