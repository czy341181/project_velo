from PIL import Image
import numpy as np
import kitti_util
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from numpy import random,mat
import argparse
import tqdm
import os


def generate_depth_from_velo(pc_velo, height, width, calib):
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < width - 1) & (pts_2d[:, 0] >= 0) & \
               (pts_2d[:, 1] < height - 1) & (pts_2d[:, 1] >= 0)
    #fov_inds = fov_inds
    fov_inds = fov_inds & (pc_velo[:, 0] > 2)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)
    depth_map = np.zeros((height, width),dtype=np.float32)
    xyz_map = np.zeros((height, width, 3),dtype=np.float32)

    imgfov_pts_2d = np.round(imgfov_pts_2d).astype(int)
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        depth_map[imgfov_pts_2d[i, 1], imgfov_pts_2d[i, 0]] = depth

        xyz_map[imgfov_pts_2d[i, 1], imgfov_pts_2d[i, 0], 2] = imgfov_pc_rect[i, 2]
        xyz_map[imgfov_pts_2d[i, 1], imgfov_pts_2d[i, 0], 0] = imgfov_pc_rect[i, 0]
        xyz_map[imgfov_pts_2d[i, 1], imgfov_pts_2d[i, 0], 1] = imgfov_pc_rect[i, 1]
    return depth_map, xyz_map



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_depth', action='store_true', help='Generate depth map')
    parser.add_argument('--gen_xyz', action='store_true', help='Generate xyz coordinate map')
    parser.add_argument('--vis_proj', action='store_true',help='Project velo to image plane for vis')
    args = parser.parse_args()


    name_list = os.listdir('/data2/czy/data/Kitti/object/training/image_2/')
    progress_bar = tqdm.tqdm(total=len(name_list), leave=True, desc='generate kitti depth xyz')
    for data_idx in name_list: # image idx
        name = data_idx

        image_2_path = '/data2/czy/data/Kitti/object/training/image_2/' + name
        image_3_path = '/data2/czy/data/Kitti/object/training/image_3/' + name
        calib_path = '/data2/czy/data/Kitti/object/training/calib/' + name.replace('.png','.txt')
        velodyne = '/data2/czy/data/Kitti/object/training/velodyne/' + name.replace('.png','.bin')

        image_2 = Image.open(image_2_path)
        width, height = image_2.size

        calib = kitti_util.Calibration(calib_path)
        lidar = np.fromfile(velodyne, dtype=np.float32).reshape((-1, 4))[:, :3]

        if args.gen_depth:
            save_path = '/data1/czy/3D/code/project_velo/save_dir/project_depth/'
        ##################project depth map######################
            depth_map, xyz_map = generate_depth_from_velo(lidar, height, width, calib)
            depth_map = np.array(depth_map*256, dtype=np.uint16)
            cv2.imwrite(save_path+name, depth_map)
            if args.gen_xyz:
                save_path = '/data1/czy/3D/code/project_velo/save_dir/project_xyz/'
                xyz = xyz_map.reshape(height, width, 3)  # record xyz, data type: float32
                np.save(save_path+name.replace('.png','.npy'), xyz)
                #print(xyz_map.shape)

            #TODO:  debug
        #     save_path = '/data1/czy/3D/code/project_velo/save_dir/project_xyz/'
        # ##################generate xyz coordinates map##################
        #     depth_map,_ = generate_depth_from_velo(lidar, height, width, calib)
        #     depth = np.array(depth_map).astype(np.float32)
        #     uvdepth = np.zeros((height, width, 3), dtype=np.float32)
        #     for v in range(height):
        #         for u in range(width):
        #             uvdepth[v, u, 0] = u
        #             uvdepth[v, u, 1] = v
        #     uvdepth[:, :, 2] = depth
        #     uvdepth = uvdepth.reshape(-1, 3)
        #     xyz = calib.img_to_rect(uvdepth[:, 0], uvdepth[:, 1], uvdepth[:, 2])  # rect coord sys
        #     xyz = xyz.reshape(height, width, 3)  # record xyz, data type: float32
        #     np.save(save_path+name.replace('.png','_1.npy'), xyz)


        if args.vis_proj:

            save_path = '/data1/czy/3D/code/project_velo/save_dir/project_lidar_vis/'
        ###################lidar project vis#####################
            with open(calib_path, 'r') as f:
                calib = f.readlines()

            P2 = np.matrix([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3, 4)
            R0_rect = np.matrix([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3, 3)
            # Add a 1 in bottom-right, reshape to 4 x 4
            R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0], axis=0)
            R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0, 1], axis=1)
            Tr_velo_to_cam = np.matrix([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3, 4)
            Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)

            scan = np.fromfile(velodyne, dtype=np.float32).reshape((-1, 4))
            points = scan[:, 0:3]  # lidar xyz (front, left, up)
            # TODO: use fov filter?
            velo = np.insert(points, 3, 1, axis=1).T
            velo = np.delete(velo, np.where(velo[0, :] < 0), axis=1)

            cam = P2 * R0_rect * Tr_velo_to_cam * velo
            cam = np.delete(cam, np.where(cam[2, :] < 0)[1], axis=1)
            # get u,v,z
            cam[:2] /= cam[2, :]
            # do projection staff
            plt.figure(figsize=(12, 5), dpi=96, tight_layout=True)
            png = mpimg.imread(image_2_path)
            IMG_H, IMG_W, _ = png.shape
            # restrict canvas in range
            plt.axis([0, IMG_W, IMG_H, 0])
            plt.imshow(png)
            # filter point out of canvas
            u, v, z = cam
            u_out = np.logical_or(u < 0, u > IMG_W)
            v_out = np.logical_or(v < 0, v > IMG_H)
            outlier = np.logical_or(u_out, v_out)
            cam = np.delete(cam, np.where(outlier), axis=1)
            u, v, z = cam
            plt.scatter([u], [v], c=[z], cmap='rainbow_r', alpha=0.5, s=2)
            plt.title(name)
            plt.savefig(save_path+name, bbox_inches='tight')
            plt.show()

        progress_bar.update()
    progress_bar.close()