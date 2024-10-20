import os
import numpy as np
import open3d as o3d
from tqdm import tqdm

def load_npy_point_cloud(file_path):
    """加载 .npy 文件并转换为 Open3D 点云格式"""
    data = np.load(file_path)
    xyz = data[:, :3]  # 提取XYZ坐标
    rgb = data[:, 3:6]  # 提取RGB颜色，并归一化到 [0, 1] 范围
    if rgb.max() > 1.0:
        rgb /= 255

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(rgb)
    
    return point_cloud

def apply_statistical_outlier_removal(pcd, nb_neighbors=20, std_ratio=2.0):
    """应用统计滤波进行去噪"""
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_cloud = pcd.select_by_index(ind)
    return inlier_cloud

def register_point_clouds(source, target, voxel_size):
    """使用ICP算法配准点云"""
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    threshold = voxel_size * 1.5
    result_icp = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    source.transform(result_icp.transformation)
    return source

def combine_point_clouds_from_folder(folder_path, voxel_size=0.05):
    """从文件夹中加载所有 .npy 文件，并配准后合成为一个点云"""
    combined_pcd = None
    
    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            # print(f"Loading {file_path}")
            current_pcd = load_npy_point_cloud(file_path)
            
            # 统计滤波去噪
            current_pcd = apply_statistical_outlier_removal(current_pcd)
            
            if combined_pcd is None:
                combined_pcd = current_pcd
            else:
                current_pcd = register_point_clouds(current_pcd, combined_pcd, voxel_size)
                combined_pcd += current_pcd
    
    return combined_pcd

def apply_uniform_sampling(pcd, number_of_points):
    """均匀采样点云（Open3D格式）并确保点数与 number_of_points 一致"""
    if isinstance(pcd, np.ndarray):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    
    original_points = np.asarray(pcd.points)
    original_colors = np.asarray(pcd.colors)
    original_count = len(original_points)
    
    if original_count >= number_of_points:
        step = original_count // number_of_points
        indices = np.arange(0, original_count, step)
        if len(indices) > number_of_points:
            indices = indices[:number_of_points]
        sampled_points = original_points[indices]
        sampled_colors = original_colors[indices]
    else:
        # 如果点数不足，通过插值和重复来填补
        sampled_points = np.empty((number_of_points, 3))
        sampled_colors = np.empty((number_of_points, 3))
        repeat_times = number_of_points // original_count
        remainder = number_of_points % original_count
        
        # 先复制完整的几遍
        for i in range(repeat_times):
            sampled_points[i*original_count:(i+1)*original_count] = original_points
            sampled_colors[i*original_count:(i+1)*original_count] = original_colors
        
        # 然后处理剩余的部分
        if remainder > 0:
            sampled_points[repeat_times*original_count:] = original_points[:remainder]
            sampled_colors[repeat_times*original_count:] = original_colors[:remainder]
    
    # 创建新的点云对象
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    sampled_pcd.colors = o3d.utility.Vector3dVector(sampled_colors)
    
    return sampled_pcd

# 文件夹路径
folder_path = "/f/pro/Hik/code/sensing3d/data/hik/00"

# 合成点云
combined_pcd = combine_point_clouds_from_folder(os.path.join(folder_path, 'point'))

# 打印合成后的点云点数
print(f"Combined point cloud has {len(combined_pcd.points)} points.")

# 均匀采样到50万个点
sampled_pcd = apply_uniform_sampling(combined_pcd, 500000)

# 打印采样后的点云点数
print(f"Sampled point cloud has {len(sampled_pcd.points)} points.")

# 保存采样后的点云为 PLY 文件
o3d.io.write_point_cloud(os.path.join(folder_path, "00.ply"), sampled_pcd)
print(f"Sampled point cloud saved!")

# 可视化采样后的点云
o3d.visualization.draw_geometries([sampled_pcd])
