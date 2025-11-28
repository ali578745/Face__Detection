import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
from utils.stream import run_inference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True,help='Image, "webcam", or video file')
    parser.add_argument('--database', type=str, required=True,help='Path to database folder')
    args = parser.parse_args()
    run_inference(args)

if __name__ == "__main__":
    main()