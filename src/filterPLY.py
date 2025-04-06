import sys
from plyfile import PlyData, PlyElement

COLOR_KEY = (0, 0, 0) # RGB

def save_ply(path_ply,Xs,Ys, Zs, Rs, Gs, Bs):

    with open(path_ply, mode='w') as f:

        line = 'ply\n'
        f.write(line)

        line = 'format ascii 1.0\n'
        f.write(line)

        line = 'element vertex %d\n' % len(Xs)
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

        for x, y, z, r, g, b in zip(Xs, Ys, Zs, Rs, Gs, Bs):
            line = '%f %f %f %d %d %d\n' % (x, y, z, r, g, b)
            f.write(line)

    print('save %s' % path_ply)

def main():

    global COLOR_KEY

    argv = sys.argv
    argc = len(argv)

    print('%s filters ply using color key' % argv[0])
    print('[usage] python %s <PLY file> [<color key(R)> <color key(G)> <color key(B)>]' % argv[0])

    if argc < 2:
        quit()

    if argc > 2:
        COLOR_KEY[0] = int(argv[2])

    if argc > 3:
        COLOR_KEY[1] = int(argv[3])

    if argc > 4:
        COLOR_KEY[2] = int(argv[4])

    plydata = PlyData.read(argv[1])
    XX = plydata.elements[0].data['x']
    YY = plydata.elements[0].data['y']
    ZZ = plydata.elements[0].data['z']
    RR = plydata.elements[0].data['red']
    GG = plydata.elements[0].data['green']
    BB = plydata.elements[0].data['blue']

    Xs = []
    Ys = []
    Zs = []
    Rs = []
    Gs = []
    Bs = []

    for i in range(XX.shape[0]):
        if RR[i] == COLOR_KEY[0] and GG[i] == COLOR_KEY[1] and BB[i] == COLOR_KEY[2]:
            continue

        Xs.append(XX[i])
        Ys.append(YY[i])
        Zs.append(ZZ[i])
        Rs.append(RR[i])
        Gs.append(GG[i])
        Bs.append(BB[i])

    save_ply('filtered.ply', Xs, Ys, Zs, Rs, Gs, Bs)

if __name__ == '__main__':
    main()
