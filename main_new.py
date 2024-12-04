from pathlib import Path
import shutil
import argparse
import numpy as np
import time
import ffmpeg
import torch
from extract_features_new import run  # Updated import for features_new.py
from models.resnet import i3_res50
import os


def load_class_names(class_names_file):
    """
    Load class names from a text file. Each line in the file corresponds to a class name.
    """
    if not class_names_file or not os.path.exists(class_names_file):
        print("Class names file not provided or does not exist.")
        return None
    with open(class_names_file, 'r') as f:
        class_names = f.read().splitlines()
    print(f"Loaded {len(class_names)} class names.")
    return class_names


def generate(datasetpath, outputpath, pretrainedpath, frequency,segment_size, batch_size, sample_mode, classify, class_names_file=None):
    Path(outputpath).mkdir(parents=True, exist_ok=True)
    temppath = os.path.join(outputpath, "temp/")
    rootdir = Path(datasetpath)
    videos = [str(f) for f in rootdir.glob('**/*.mp4')]

    # Setup the model
    i3d = i3_res50(num_classes=400,pretrainedpath=pretrainedpath,classify=classify)
    i3d.cuda()
    i3d.eval()  # Set model to evaluate mode

    # Load class names if classification mode is enabled
    class_names = load_class_names(class_names_file) if classify else None

    for video in videos:
        videoname = video.split("/")[-1].split(".")[0]
        start_time = time.time()
        print("Processing video:", video)
        Path(temppath).mkdir(parents=True, exist_ok=True)

        # Extract frames using FFmpeg
        ffmpeg.input(video).output(
            f'{temppath}%d.jpg',
            vf='scale=640:-1',
            start_number=0
        ).global_args('-loglevel', 'quiet').run()
        print("Preprocessing done...")

        # Call the feature extraction and classification function
        run(
            i3d=i3d,
            classify=classify,
            frequency=frequency,
            segment_size=segment_size,
            frames_dir=temppath,
            batch_size=batch_size,
            sample_mode=sample_mode,
            class_names=class_names,
            video_id=os.path.join(outputpath, videoname)
        )

        # Clean up temporary directory
        shutil.rmtree(temppath)
        print(f"Processed {video} in {time.time() - start_time:.2f} seconds.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetpath', type=str, default="output_classes", help="Path to the folder containing input videos.")
    parser.add_argument('--outputpath', type=str, default="output_classes/features", help="Path to the folder where outputs will be saved.")
    parser.add_argument('--pretrainedpath', type=str, default="pretrained/i3d_r50_kinetics.pth", help="Path to the pretrained model weights.")
    parser.add_argument('--frequency', type=int, default=16, help="Frequency of frame sampling.")
    parser.add_argument('--batch_size', type=int, default=20, help="Batch size for processing.")
    parser.add_argument('--segment_size',type=int,default=16,help='Frames count within a segment or clip')
    parser.add_argument('--sample_mode', type=str, default="center_crop", help="Sample mode: 'center_crop' or 'oversample'.")
    parser.add_argument('--classify', action='store_true', help="Flag to enable classification mode.")
    parser.add_argument('--class_names_file', type=str, default='kinetics_400_classes.txt', help="Path to the text file containing class names (one per line).")
    args = parser.parse_args()

    generate(
        datasetpath=args.datasetpath,
        outputpath=args.outputpath,
        pretrainedpath=args.pretrainedpath,
        frequency=args.frequency,
        segment_size=args.segment_size,
        batch_size=args.batch_size,
        sample_mode=args.sample_mode,
        classify=args.classify,
        class_names_file=args.class_names_file
    )
                                                                     