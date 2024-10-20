import json
import os
import cv2
import numpy as np
from tqdm import tqdm

def load_camera_calibration(calib_file):
    with open(calib_file, 'r') as f:
        calib_data = json.load(f)
        
    intrinsic_matrix = np.array(calib_data['intrinsic_matrix']).reshape(3, 3)
    distortion_coeffs = np.array(calib_data['distortion_coefficients'])
    
    return intrinsic_matrix, distortion_coeffs

def find_pose(depth_images, rgb_images, camera_intrinsics, distortion_coeffs):
    poses = []
    
    # Create a feature detector and descriptor extractor
    orb = cv2.ORB_create()
    
    # Iterate over each pair of consecutive images
    for i in range(1, len(depth_images)):
        # Detect keypoints and extract descriptors
        keypoints1, descriptors1 = orb.detectAndCompute(rgb_images[i-1], None)
        keypoints2, descriptors2 = orb.detectAndCompute(rgb_images[i], None)
        
        # Match descriptors using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        
        # Sort the matches based on distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract matched keypoints
        points1 = np.array([keypoints1[m.queryIdx].pt for m in matches], dtype=np.float32)
        points2 = np.array([keypoints2[m.trainIdx].pt for m in matches], dtype=np.float32)
        
        # Use depth information to get 3D points
        points3D_1 = []
        for pt in points1:
            z = depth_images[i-1][int(pt[1]), int(pt[0])]
            x = (pt[0] - camera_intrinsics[0, 2]) * z / camera_intrinsics[0, 0]
            y = (pt[1] - camera_intrinsics[1, 2]) * z / camera_intrinsics[1, 1]
            points3D_1.append([x, y, z])
        points3D_1 = np.array(points3D_1, dtype=np.float32)
        
        # Solve PnP to get pose
        _, rvec, tvec = cv2.solvePnP(points3D_1, points2, camera_intrinsics, distortion_coeffs)
        
        # Convert rotation vector to matrix
        rot_matrix, _ = cv2.Rodrigues(rvec)
        
        # Form the pose matrix (4x4)
        pose = np.eye(4)
        pose[:3, :3] = rot_matrix
        pose[:3, 3] = tvec.flatten()
        
        poses.append(pose)
    
    return poses

def load_data(dir, file_extension):
    name = sorted(os.listdir(dir))
    path = [os.path.join(dir, p) for p in name if p.endswith(file_extension)]
    images = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in path]
    return images

def save_poses(pose_dir, depth_dir, poses):
    depth_files = sorted(os.listdir(depth_dir))
    num_poses = len(poses)
    for i in tqdm(range(num_poses)):
        pose_filename = os.path.join(pose_dir, depth_files[i][:-4] + '.npy')
        np.save(pose_filename, poses[i])

depth_dir = '/f/pro/Hik/code/sensing3d/data/hik/00/depth'
depth_images = load_data(depth_dir, '.png')

image_dir = '/f/pro/Hik/code/sensing3d/data/hik/00/image'
rgb_images = load_data(image_dir, '.png')

calibration_file = 'rgb_calib.json'
camera_intrinsics, distortion_coeffs = load_camera_calibration(calibration_file)

pose_dir = '/f/pro/Hik/code/sensing3d/data/hik/00/pose'
poses = find_pose(depth_images, rgb_images, camera_intrinsics, distortion_coeffs)
save_poses(pose_dir, depth_dir, poses)
# for i, pose in enumerate(poses):
#     print(f"Pose for image {i+1} saved.")
