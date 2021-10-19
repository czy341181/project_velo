import numpy as np
from PIL import Image
import kitti_util


xyz = np.load('/data1/czy/3D/code/project_velo/save_dir/project_xyz/000199.npy')
#print(xyz.shape)
h, w, c = xyz.shape

depth = Image.open('/data1/czy/3D/code/project_velo/save_dir/project_depth/000199.png')
width, height = depth.size
depth = np.array(depth)/256

velo = '/data2/czy/data/Kitti/object/training/velodyne/000199.bin'
calib_path = '/data2/czy/data/Kitti/object/training/calib/000199.txt'

lidar = np.fromfile(velo, dtype=np.float32).reshape((-1, 4))[:, :3]
calib = kitti_util.Calibration(calib_path)


pts_2d = calib.project_velo_to_image(lidar)
fov_inds = (pts_2d[:, 0] < width - 1) & (pts_2d[:, 0] >= 0) & \
           (pts_2d[:, 1] < height - 1) & (pts_2d[:, 1] >= 0)
fov_inds = fov_inds & (lidar[:, 0] > 2)
imgfov_pc_velo = lidar[fov_inds, :]
imgfov_pts_2d = pts_2d[fov_inds, :]
imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)
print("imgfov_pc_rect:",imgfov_pc_rect.shape)
print("imgfov_pts_2d:",imgfov_pts_2d.shape)
num_point, _ = imgfov_pc_velo.shape


print(imgfov_pc_rect[:,1].max())
print(imgfov_pc_rect[:,1].min())
# for i in range(num_point):
#     print(imgfov_pc_velo[i,1])

#print(jjj)
#print(depth.shape)
sum0 = 0
sum1 = 0
max = -80
min = 80
for i in range(h):
    for j in range(w):
        if xyz[i,j,1]>=max:
    #             sum0 = sum0 + 1
            max = xyz[i,j,1]
        if xyz[i,j,1]<=min:
            min = xyz[i,j,1]
print("max:",max)
print("min:",min)
            #print(xyz[i,j,2])
# print("sum0:",sum0)
#print("sum1:",sum1)
