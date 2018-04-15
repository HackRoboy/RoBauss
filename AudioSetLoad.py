ffmpeg_path = '/usr/bin/ffmpeg'

import sys
import os.path
# Make sure ffmpeg is on the path so sk-video can find it
sys.path.append(os.path.dirname(ffmpeg_path))
import skvideo.io
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pafy
import soundfile as sf
import subprocess as sp
import random

# Set output settings
audio_codec = 'wav'
audio_container = 'wav'
video_codec = 'h264'
video_container = 'mp4'

# Load the AudioSet training set
with open('data/balanced_train_segments.csv') as f:
    lines = f.readlines()

dl_list = [[s.replace('\"', '').replace(' ','') for s in line.strip().split(',')] for line in lines[3:]]

# Load the AudioSet class names
with open('data/class_labels_indices.csv') as f:
    lines = f.readlines()

cl_list = [line.strip().split(',')[0:3] for line in lines[1:]]

index_dictionary = {b : c for a,b,c in cl_list}

file_labelling = {a[0]:[index_dictionary[c] for c in a[3:]] for a in dl_list}

reverse_index_dictionary = {c:b for a,b,c in cl_list}

def dlfile(ytid, ts_start, ts_end):
    # Set output settings
    audio_codec = 'wav'
    audio_container = 'wav'
    video_codec = 'h264'
    video_container = 'mp4'
    ts_start, ts_end = float(ts_start), float(ts_end)
    duration = ts_end - ts_start

    # Get output video and audio filepaths
    basename_fmt = '{}_{}_{}'.format(ytid, int(ts_start*1000), int(ts_end*1000))
    audio_filepath = os.path.join('.', basename_fmt + '.' + audio_codec)
    # Download the audio
    
    # Get the URL to the video page
    video_page_url = 'https://www.youtube.com/watch?v={}'.format(ytid)

    # Get the direct URLs to the videos with best audio and with best video (with audio)
    video = pafy.new(video_page_url)
    
    best_audio = video.getbestaudio()
    best_audio_url = best_audio.url
    
    audio_dl_args = [ffmpeg_path, 
    '-ss', str(ts_start),    # The beginning of the trim window
    '-i', best_audio_url,    # Specify the input video URL
    '-t', str(duration),     # Specify the duration of the output
    '-vn',                   # Suppress the video stream
    '-ac', '2',              # Set the number of channels
    '-y',                    # overwrite
    '-sample_fmt', 's16',    # Specify the bit depth
    #'-acodec', audio_codec,  # Specify the output encoding
    '-ar', '44100',          # Specify the audio sample rate
    audio_filepath]

    proc = sp.Popen(audio_dl_args, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        print(stderr)
    return audio_filepath

def dl_random_file():
	ytid, ts_start, ts_end = random.choice(dl_list)[:3]
	return dlfile(ytid, ts_start, ts_end), ytid, file_labelling[ytid]

