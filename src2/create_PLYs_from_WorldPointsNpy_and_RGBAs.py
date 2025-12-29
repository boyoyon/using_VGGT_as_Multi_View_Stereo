import cv2, glob, os, sys
import numpy as np

def save_ply(path_ply, lines):

    with open(path_ply, mode='w') as f:

        line = 'ply\n'
        f.write(line)

        line = 'format ascii 1.0\n'
        f.write(line)

        line = 'element vertex %d\n' % len(lines)
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

        for line in lines:
            f.write(line)

    print('save %s' % path_ply)

def main():

    maskTH = 128

    argv = sys.argv
    argc = len(argv)
    
    print('%s creates PLYs from WorldPoints.npy and RGBA images' % argv[0])
    print('[usage] python %s <WorldPoints.npy> <wildcard for RGBA images>' % argv[0])
    
    if argc < 3:
        quit()
   
    world_points = np.load(argv[1])
    H = world_points.shape[1]
    W = world_points.shape[2]

    RGBA_names = glob.glob(argv[2])
    nrRGBAs = len(RGBA_names)

    if world_points.shape[0] != nrRGBAs:
        print('number of world_points is mismatch with the number of RGBAs', world_points.shape[0], len(RGBA_names))
        quit()


    for i in range(nrRGBAs):

        lines = []

        RGBA = cv2.imread(RGBA_names[i], cv2.IMREAD_UNCHANGED)

        if RGBA.shape[0] != H or RGBA.shape[1] != W:
            RGBA = cv2.resize(RGBA, (W, H))

        B, G, R, mask = cv2.split(RGBA)

        for Y in range(H):

            for X in range(W):

                if mask[Y][X] < maskTH:
                    continue

                x = float(world_points[i][Y][X][0])
                y = float(world_points[i][Y][X][1])
                z = float(world_points[i][Y][X][2])
                r = R[Y][X]
                g = G[Y][X]
                b = B[Y][X]

                line = '%f %f %f %d %d %d\n' % (x, y, z, r, g, b)

                lines.append(line)

        dst_ply = '%04d.ply' % (i+1)
        save_ply(dst_ply, lines)

if __name__ == '__main__':
    main()
