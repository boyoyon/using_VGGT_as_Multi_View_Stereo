import open3d as o3d
import numpy as np
import os, sys

argv = sys.argv
argc = len(argv)

print('%s merges ply1 and ply2' % argv[0])
print('[usage] python %s <ply1> <ply2>' % argv[0])

if argc < 3:
    quit()

# PLYファイルを読み込む
source = o3d.io.read_point_cloud(argv[1])  

base1 = os.path.basename(argv[1])
filename1 = os.path.splitext(base1)[0]

target = o3d.io.read_point_cloud(argv[2])  

base2 = os.path.basename(argv[2])
filename2 = os.path.splitext(base2)[0]

# 外れ値除去
source = source.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]
target = target.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]

threshold = 0.02  # 近傍探索の閾値
trans_init = np.eye(4)  # 初期変換行列（単位行列）

# ICPの実行
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
)

print("ICP変換行列:")
print(reg_p2p.transformation)
# 参照用点群に対して変換行列により剛体変換
source.transform(reg_p2p.transformation)

# 変換後の点群を保存
dst_path = 'ICP_%s_%s.ply' % (filename1, filename2)
merged = source + target
o3d.io.write_point_cloud(dst_path, merged)
print('save %s' % dst_path)

# 変換後の点群を可視化
o3d.visualization.draw_geometries([source, target])