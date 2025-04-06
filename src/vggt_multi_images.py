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

def save_ply(path_ply, world_points, image_names):

    nrImages, H, W = world_points.shape[:3]

    if nrImages != len(image_names):
       print('nrImages mismatch: world_points(%d), image_nmaes(%d)' % (nrImages, len(image_names)))
       return

    images = []
    for i in range(nrImages):
       img = cv2.imread(image_names[i])
       img = cv2.resize(img, (W, H))
       images.append(img)

    with open(path_ply, mode='w') as f:

        line = 'ply\n'
        f.write(line)

        line = 'format ascii 1.0\n'
        f.write(line)

        line = 'element vertex %d\n' % (nrImages * H * W)
        f.write(line)

        line = 'property float x\n'
        f.write(line)

        line = 'property float y\n'
        f.write(line)

        line = 'property float z\n'
        f.write(line)

        line = 'property uchar red\n'
        f.write(line)

        line = 'property uchar green\n'
        f.write(line)

        line = 'property uchar blue\n'
        f.write(line)

        line = 'end_header\n'
        f.write(line)

        for i in range(nrImages):
           for Y in range(H):
               for X in range(W):

                   x = float(world_points[i][Y][X][0])
                   y = float(world_points[i][Y][X][1])
                   z = float(world_points[i][Y][X][2])
                   r = images[i][Y][X][2]
                   g = images[i][Y][X][1]
                   b = images[i][Y][X][0]

                   line = '%f %f %f %d %d %d\n' % (x, y, z, r, g, b)            

                   f.write(line)

    print('save %s' % path_ply)

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

    save_ply('world_points.ply', world_points, image_names)

if __name__ == '__main__':
    main()
