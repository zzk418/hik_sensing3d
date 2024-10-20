import os
import numpy as np
import cv2
import json
import open3d as o3d

def load_camera_calibration(calibration_file):
    with open(calibration_file, 'r') as f:
        calib_data = json.load(f)
    camera_intrinsics = np.array(calib_data['intrinsic_matrix']).reshape(3, 3)
    distortion_coeffs = np.array(calib_data['distortion_coefficients'])
    return camera_intrinsics, distortion_coeffs

def adjust_intrinsic(intrinsics, original_dim, new_dim):
    scale_x = new_dim[0] / original_dim[0]
    scale_y = new_dim[1] / original_dim[1]
    adjusted_intrinsics = intrinsics.copy()
    adjusted_intrinsics[0, 0] *= scale_x
    adjusted_intrinsics[1, 1] *= scale_y
    adjusted_intrinsics[0, 2] *= scale_x
    adjusted_intrinsics[1, 2] *= scale_y
    return adjusted_intrinsics

def load_data(directory, ext):
    images = []
    names = []
    for filename in os.listdir(directory):
        if filename.endswith(ext):
            img_path = os.path.join(directory, filename)
            images.append(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
            names.append(filename)
    return images, names

def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)

def create_point_cloud(depth_image, color_image, intrinsics):
    height, width = depth_image.shape
    point_cloud = []

    for v in range(height):
        for u in range(width):
            Z = depth_image[v, u]  # 深度值
            if Z == 0:  # 忽略无效深度
                continue
            # 根据相机内参计算点云坐标
            X = (u - intrinsics[0, 2]) * Z / intrinsics[0, 0]
            Y = (v - intrinsics[1, 2]) * Z / intrinsics[1, 1]
            point_cloud.append([X, Y, Z, color_image[v, u, 0], color_image[v, u, 1], color_image[v, u, 2]])

    return np.array(point_cloud)

def save_point_cloud(point_cloud, filename):
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6] / 255.0)  # 归一化颜色值

    # 下采样
    voxel_size = 0.05  # 您可以根据需要调整体素大小
    pcd = pcd.voxel_down_sample(voxel_size)

    # 估计法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 统计离群点移除
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)

    # 将处理后的点云数据保存为 .npy 文件
    # 获取点和颜色
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    
    # 合并点和颜色数据
    processed_point_cloud = np.hstack((points, colors))

    # 保存为 .npy 文件
    np.save(filename, processed_point_cloud)

if __name__ == '__main__':
    calibration_file = 'rgb_calib.json'
    camera_intrinsics, distortion_coeffs = load_camera_calibration(calibration_file)
    unify_dim = (640, 480)
    unify_intrinsic = adjust_intrinsic(camera_intrinsics, [1280, 720], unify_dim)

    depth_dir = '/f/pro/Hik/code/sensing3d/data/hik/00/depth'
    color_dir = '/f/pro/Hik/code/sensing3d/data/hik/00/image'
    depth_images, depth_names = load_data(depth_dir, '.png')
    color_images, _ = load_data(color_dir, '.png')

    save_dir = '/f/pro/Hik/code/sensing3d/data/hik/00'
    point_dir = os.path.join(save_dir, 'point')
    os.makedirs(point_dir, exist_ok=True)
    clear_folder(point_dir)

    for depth_image, color_image, depth_name in zip(depth_images, color_images, depth_names):
        point_cloud = create_point_cloud(depth_image, color_image, unify_intrinsic)
        point_cloud_filename = os.path.join(point_dir, f'{depth_name}.npy')
        save_point_cloud(point_cloud, point_cloud_filename)
