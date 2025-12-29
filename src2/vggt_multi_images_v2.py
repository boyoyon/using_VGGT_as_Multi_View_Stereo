import cv2, glob, os, sys
import torch
import numpy as np
import gradio as gr
import shutil
from datetime import datetime
import gc
import time

sys.path.append("vggt/")

#from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

def main():

    argv = sys.argv
    argc = len(argv)
    
    print('%s executes VGGT prediction' % argv[0])
    print('[usage] python %s <wildcard for images>' % argv[0])
    print('[usage] python %s <image1> <image2> ...' % argv[0])
    
    if argc < 2:
        quit()
    
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    base = os.path.basename(argv[1])
    filename = os.path.splitext(base)[0]

    image_names = None

    if argc > 2:
        image_names = []
        for i in range(1, argc):
            image_names.append(argv[i])

    else:
        image_names = glob.glob(argv[1])        

    print(image_names)
 
    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")  # another way to load the model
    
    model = VGGT()
    model.load_state_dict(torch.load('model.pt', map_location=torch.device(device)))
    
    model = model.to(device)
    model.eval()
    
    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")
    
    # Run inference
    print("Running inference...")
    #dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    dtype = torch.float32

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])

    # Clean up
    torch.cuda.empty_cache()

    np.save('world_points.npy', world_points)
    print('save world_points.npy')

    np.save('intrinsic.npy', predictions["intrinsic"])
    print('save intrinsic.npy')

    np.save('extrinsic.npy', predictions["extrinsic"])
    print('save extrinsic.npy')

if __name__ == '__main__':
    main()
