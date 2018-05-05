import cv2
import glob
import os
import logging
import pkg_resources

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
    import frames
    # frames directory name will correspond to video name
    # will write to absolute or relative paths provided creating corresponding directories
    - to_frames(input_video_file_path, output_frames_directory_path)
    - video_directory_to_frames(input_video_directory_path, output_frames_directories_directory_path)
"""

def to_frames(input_video_file_path, output_frames_parent_directory_path, directory_name="DEFAULT"):
    """
    Currently takes only absolute paths
    Splits input video into directory containing frames labeled 0.jpg,..,n.jpg. Directory name corresponds to video name
    :param input_video_file_path: Path of video to split into frames.
    :param output_frames_parent_directory_path: Path to write output frames directory to. Note: Name of \
        directory, directory name corresponds to video name by default
    :param directory_name: "DEFAULT" by default will correspond to video file name, None will insert videos directly \
        into parent directory path, any other valid string will be used as directory name
    "
    """
    assert os.path.isfile(input_video_file_path)

    # directory name
    frames_directory = output_frames_parent_directory_path
    if directory_name == "DEFAULT":
        video_file_name = os.path.basename(input_video_file_path).split('.')[0]
        frames_directory = os.path.join(frames_directory, video_file_name)
    elif directory_name != None:
        frames_directory = os.path.join(frames_directory, directory_name)

    # create resource/frame directory for video if it does not exist
    logger.info("\n\tCreating frames directory...\n\tInput Video: %s\n\tOutput Frames Directory: %s" % (
        input_video_file_path, frames_directory))

    if not os.path.exists(frames_directory):
        os.makedirs(frames_directory)

    # individual frame file name template
    frame_i_file_path_template = "%d.jpg"

    # write frame jpegs
    logger.info('\n\tWriting frames...')
    video_cap = cv2.VideoCapture(input_video_file_path)
    success, image = video_cap.read()
    count = 0
    while success:
        if success:
            output_file = frame_i_file_path_template % count
            output_file_path = os.path.join(frames_directory, output_file)
            logger.debug('Writing a new frame: %s ' % os.path.join(frames_directory, output_file))
            cv2.imwrite(output_file_path, image)  # save frame as JPEG file
            count += 1
        success, image = video_cap.read()
    logger.info('Finished writing frames.')


def video_directory_to_frames(input_video_directory_path, output_frames_directories_directory_path):
    """
    Currently takes only absolute paths
    For each video in input directory,
        Splits input video into directory containing frames labeled 0.jpg,..,n.jpg. Directory name corresponds to video name
    :param input_video_directory_path: Path of directory of videos to split into directories of frames.
    :param output_frames_directories_directory_path: Path to write output frames directories to. Note: Don't include name of \
        directory, directory name corresponds to specific video name
    """
    assert os.path.exists(input_video_directory_path)
    if input_video_directory_path[-1] is not '/':
        input_video_directory_path += '/'
    for video_file_path in glob.glob(input_video_directory_path + '*.mp4'):
        to_frames(video_file_path, output_frames_directories_directory_path)