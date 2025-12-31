import cv2, os, sys
import numpy as np
import open3d as o3d

KEY_LEFT  = 263
KEY_RIGHT = 262
KEY_UP    = 265
KEY_DOWN  = 264

angle_step = np.pi / 180
translation_step = 0.005
scale_up = 1.5
scale_down = 0.5

pcd = None

def key_callback_updown_angle_step(vis, action, mods):

    global angle_step, translation_step

    shift_pressed = (mods & 0x1) != 0
    ctrl_pressed = (mods & 0x2) != 0

    #if action == 1: # on pressing

    if shift_pressed:

        if ctrl_pressed:
            
            angle_step *= 1.5

        else:

            angle_step *= 1.1
    else:

        if ctrl_pressed:
            
            angle_step *= 0.5

        else:

            angle_step *= 0.9

    print(angle_step, translation_step)

    return True

def key_callback_updown_translation_step(vis, action, mods):

    global angle_step, translation_step

    shift_pressed = (mods & 0x1) != 0
    ctrl_pressed = (mods & 0x2) != 0

    #if action == 1: # on pressing

    if shift_pressed:

        if ctrl_pressed:
            
            translation_step *= 1.5

        else:

            translation_step *= 1.1
    else:

        if ctrl_pressed:
            
            translation_step *= 0.5

        else:

            translation_step *= 0.9

    print(angle_step, translation_step)

    return True

def key_callback_1(vis, action, mods):

    shift_pressed = (mods & 0x1) != 0
    ctrl_pressed = (mods & 0x2) != 0

    #if action == 1: # on pressing

    if shift_pressed:
        angle = -angle_step
    else:
        angle = angle_step

    if ctrl_pressed:
        angle *= 10

    rotation = np.array([[np.cos(angle), 0, np.sin(angle), 0],
        [0, 1, 0, 0],
        [-np.sin(angle), 0, np.cos(angle), 0],
        [0, 0, 0, 1]])

    transform = rotation #@ transform
    pcd.transform(transform)

    return True

def key_callback_2(vis, action, mods):

    shift_pressed = (mods & 0x1) != 0
    ctrl_pressed = (mods & 0x2) != 0

    #if action == 1: # on pressing

    if shift_pressed:
        angle = -angle_step
    else:
        angle = angle_step

    if ctrl_pressed:
        angle *= 10

    rotation = np.array([[1, 0, 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0],
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1]])

    transform = rotation #@ transform
    pcd.transform(transform)

    return True

def key_callback_3(vis, action, mods):

    shift_pressed = (mods & 0x1) != 0
    ctrl_pressed = (mods & 0x2) != 0

    #if action == 1: # on pressing

    if shift_pressed:
        angle = -angle_step
    else:
        angle = angle_step

    if ctrl_pressed:
        angle *= 10

    rotation = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    transform = rotation #@ transform
    pcd.transform(transform)

    return True

def key_callback_4(vis, action, mods):

    shift_pressed = (mods & 0x1) != 0
    ctrl_pressed = (mods & 0x2) != 0

    #if action == 1: # on pressing

    if shift_pressed:
        offset = -translation_step
    else:
        offset = translation_step

    if ctrl_pressed:
        offset *= 10

    translate = np.array([[1, 0, 0, offset],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    transform = translate
    pcd.transform(transform)

    return True

def key_callback_5(vis, action, mods):

    shift_pressed = (mods & 0x1) != 0
    ctrl_pressed = (mods & 0x2) != 0

    #if action == 1: # on pressing

    if shift_pressed:
        offset = -translation_step
    else:
        offset = translation_step

    if ctrl_pressed:
        offset *= 10

    translate = np.array([[1, 0, 0, 0],
        [0, 1, 0, offset],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    transform = translate
    pcd.transform(transform)

    return True

def key_callback_6(vis, action, mods):

    shift_pressed = (mods & 0x1) != 0
    ctrl_pressed = (mods & 0x2) != 0

    #if action == 1: # on pressing

    if shift_pressed:
        offset = -translation_step
    else:
        offset = translation_step

    if ctrl_pressed:
        offset *= 10

    translate = np.array([[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, offset],
        [0, 0, 0, 1]])

    transform = translate
    pcd.transform(transform)

    return True

def key_callback_reset_step(vis, action, mod):

    global angle_step, translation_step, scale

    angle_step = np.pi / 180
    translation_step = 0.005
    scale_up = 1.3
    scale_down = 0.7

    return True
    
def key_callback_scale_up(vis, action, mod):

    scale = np.array([
        [scale_up, 0,        0,        0],
        [0,        scale_up, 0,        0],
        [0,        0,        scale_up, 0],
        [0,        0,        0,        1]])

    transform = scale
    pcd.transform(transform)

    center = pcd.get_center()
    pcd.translate(-center)
    
    return True


def key_callback_scale_down(vis, action, mod):

    scale = np.array([
        [scale_down, 0,          0,          0],
        [0,          scale_down, 0,          0],
        [0,          0,          scale_down, 0],
        [0,          0,          0,          1]])

    transform = scale
    pcd.transform(transform)

    center = pcd.get_center()
    pcd.translate(-center)
    
    return True

def main():

    global pcd

    argv = sys.argv
    argc = len(argv)

    print('%s loads ply and visualizes 3d model' % argv[0])
    print('[usage] python %s <ply file> [<zscale>]' % argv[0])

    if argc < 2:
        quit()

    pcd = o3d.io.read_point_cloud(argv[1])

    if argc > 2:
        zscale = float(argv[2])

        # 点の座標をNumPy配列として取得
        # pointsは (N, 3) の配列。列は [x, y, z]
        points = np.asarray(pcd.points)

        # Z座標（インデックス2）をスケーリング
        points[:, 2] *= zscale

        # 加工した配列を点群オブジェクトに戻す
        pcd.points = o3d.utility.Vector3dVector(points)

    center = pcd.get_center()
    pcd.translate(-center)

    # 可視化の設定
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.register_key_action_callback(ord("0"), key_callback_reset_step)
    vis.register_key_action_callback(ord("1"), key_callback_1)
    vis.register_key_action_callback(ord("2"), key_callback_2)
    vis.register_key_action_callback(ord("3"), key_callback_3)
    vis.register_key_action_callback(ord("4"), key_callback_4)
    vis.register_key_action_callback(ord("5"), key_callback_5)
    vis.register_key_action_callback(ord("6"), key_callback_6)
    vis.register_key_action_callback(ord("7"), key_callback_updown_angle_step)
    vis.register_key_action_callback(ord("8"), key_callback_updown_translation_step)
    vis.register_key_action_callback(KEY_UP, key_callback_scale_up)
    vis.register_key_action_callback(KEY_DOWN, key_callback_scale_down)

    # 実行
    vis.run()
    vis.destroy_window()

    if argc > 2:
        dst_path = 'displayed.ply'
        no = 2
        while os.path.exists(dst_path):
            dst_path = 'displayed_%d.ply' % no
            no += 1

        o3d.io.write_point_cloud(dst_path, pcd)
        print('save %s' % dst_path)

if __name__ == '__main__':
    main()
