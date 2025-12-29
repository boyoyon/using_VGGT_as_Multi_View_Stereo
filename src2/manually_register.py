import cv2, os, sys
import numpy as np
import open3d as o3d

delta = np.pi / 360
delta2 = 0.0025

pcd = None

def key_callback_up(vis, action, mods):

    global delta, delta2

    shift_pressed = (mods & 0x1) != 0
    ctrl_pressed = (mods & 0x2) != 0

    #if action == 1: # on pressing

    if shift_pressed:

        if ctrl_pressed:
            
            delta2 *= 1.5

        else:

            delta2 *= 1.1
    else:

        if ctrl_pressed:
            
            delta *= 1.5

        else:

            delta *= 1.1

    print(delta, delta2)

    return True

def key_callback_down(vis, action, mods):

    global delta, delta2

    shift_pressed = (mods & 0x1) != 0
    ctrl_pressed = (mods & 0x2) != 0

    #if action == 1: # on pressing

    if shift_pressed:

        if ctrl_pressed:
            
            delta2 *= 0.5

        else:

            delta2 *= 0.9
    else:

        if ctrl_pressed:
            
            delta *= 0.5

        else:

            delta *= 0.9

    print(delta, delta2)

    return True

def key_callback_1(vis, action, mods):

    shift_pressed = (mods & 0x1) != 0

    #if action == 1: # on pressing

    if shift_pressed:
        angle = -delta
    else:
        angle = delta

    rotation = np.array([[np.cos(angle), 0, np.sin(angle), 0],
        [0, 1, 0, 0],
        [-np.sin(angle), 0, np.cos(angle), 0],
        [0, 0, 0, 1]])

    transform = rotation #@ transform
    pcd.transform(transform)

    return True

def key_callback_2(vis, action, mods):

    shift_pressed = (mods & 0x1) != 0

    #if action == 1: # on pressing

    if shift_pressed:
        angle = -delta
    else:
        angle = delta

    rotation = np.array([[1, 0, 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0],
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1]])

    transform = rotation #@ transform
    pcd.transform(transform)

    return True

def key_callback_3(vis, action, mods):

    shift_pressed = (mods & 0x1) != 0

    #if action == 1: # on pressing

    if shift_pressed:
        angle = -delta
    else:
        angle = delta

    rotation = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    transform = rotation #@ transform
    pcd.transform(transform)

    return True

def key_callback_4(vis, action, mods):

    shift_pressed = (mods & 0x1) != 0

    #if action == 1: # on pressing

    if shift_pressed:
        offset = -delta2
    else:
        offset = delta2

    translate = np.array([[1, 0, 0, offset],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    transform = translate
    pcd.transform(transform)

    return True

def key_callback_5(vis, action, mods):

    shift_pressed = (mods & 0x1) != 0

    #if action == 1: # on pressing

    if shift_pressed:
        offset = -delta2
    else:
        offset = delta2

    translate = np.array([[1, 0, 0, 0],
        [0, 1, 0, offset],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    transform = translate
    pcd.transform(transform)

    return True

def key_callback_6(vis, action, mods):

    shift_pressed = (mods & 0x1) != 0

    #if action == 1: # on pressing

    if shift_pressed:
        offset = -delta2
    else:
        offset = delta2

    translate = np.array([[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, offset],
        [0, 0, 0, 1]])

    transform = translate
    pcd.transform(transform)

    return True

def main():

    global pcd

    argv = sys.argv
    argc = len(argv)

    print('%s loads ply and visualizes 3d model' % argv[0])
    print('%s <ply1> <ply2>' % argv[0])

    if argc < 3:
        quit()


    pcd0 = o3d.io.read_point_cloud(argv[1])
    center0 = pcd0.get_center()
    pcd0.translate(-center0)

    pcd = o3d.io.read_point_cloud(argv[2])
    center = pcd.get_center()
    pcd.translate(-center)

    # 可視化の設定
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(pcd0)
    vis.add_geometry(pcd)
    vis.register_key_action_callback(ord("1"), key_callback_1)
    vis.register_key_action_callback(ord("2"), key_callback_2)
    vis.register_key_action_callback(ord("3"), key_callback_3)
    vis.register_key_action_callback(ord("4"), key_callback_4)
    vis.register_key_action_callback(ord("5"), key_callback_5)
    vis.register_key_action_callback(ord("6"), key_callback_6)
    vis.register_key_action_callback(ord("7"), key_callback_up)
    vis.register_key_action_callback(ord("8"), key_callback_down)

    # 実行
    vis.run()
    vis.destroy_window()

    o3d.io.write_point_cloud('merged.ply', pcd0 + pcd)

if __name__ == '__main__':
    main()
