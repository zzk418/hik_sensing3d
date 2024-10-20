# -- coding: utf-8 --
import shutil
import threading
import ctypes
import time
import os

import cv2
import numpy as np
from Mv3dRgbdImport.Mv3dRgbdApi import *
from Mv3dRgbdImport.Mv3dRgbdDefine import *
from Mv3dRgbdImport.Mv3dRgbdDefine import DeviceType_Ethernet, DeviceType_USB, MV3D_RGBD_FLOAT_EXPOSURETIME, \
    ParamType_Enum, ParamType_Int, CoordinateType_Depth, MV3D_RGBD_FLOAT_Z_UNIT

def parse_data(data, lens, dtype):
    c_data = ctypes.string_at(data, lens)
    res = np.frombuffer(c_data, dtype)
    return res

def normalize_point_cloud(point_cloud, range_min=-1, range_max=1):
    # 计算每个轴的最小值和最大值
    min_vals = np.min(point_cloud, axis=0)
    max_vals = np.max(point_cloud, axis=0)
    
    # 计算每个轴的范围
    scales = (max_vals - min_vals) / 2.0
    centers = (min_vals + max_vals) / 2.0
    
    # 归一化坐标，独立处理每个轴
    normalized_point_cloud = (point_cloud - centers) / scales
    
    # 缩放到所需范围
    normalized_point_cloud = normalized_point_cloud * (range_max - range_min) / 2.0 + (range_max + range_min) / 2.0
    
    return normalized_point_cloud

def apply_uniform_sampling(pcd, number_of_points):
    """均匀采样点云（NumPy格式）"""
    if number_of_points >= pcd.shape[0]:
        print("返回原点云")
        return pcd
    
    step = pcd.shape[0] // number_of_points
    indices = np.arange(0, pcd.shape[0], step)
    if len(indices) > number_of_points:
        indices = indices[:number_of_points]
    sampled_pcd = pcd[indices]
    return sampled_pcd

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)  
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)  # 递归删除目录

if __name__ == "__main__":
    sample_time = 3
    nDeviceNum=ctypes.c_uint(0)
    nDeviceNum_p=byref(nDeviceNum)
    ret=Mv3dRgbd.MV3D_RGBD_GetDeviceNumber(DeviceType_Ethernet | DeviceType_USB, nDeviceNum_p) #获取设备数量
    if  ret!=0:
        print("MV3D_RGBD_GetDeviceNumber fail! ret[0x%x]" % ret)
        os.system('pause')
        sys.exit()
    if  nDeviceNum==0:
        print("find no device!")
        os.system('pause')
        sys.exit()
    print("Find devices numbers:", nDeviceNum.value)

    stDeviceList = MV3D_RGBD_DEVICE_INFO_LIST()
    net = Mv3dRgbd.MV3D_RGBD_GetDeviceList(DeviceType_Ethernet | DeviceType_USB, pointer(stDeviceList.DeviceInfo[0]), 20, nDeviceNum_p)
    for i in range(0, nDeviceNum.value):
        print("\ndevice: [%d]" % i)
        strModeName = ""
        for per in stDeviceList.DeviceInfo[i].chModelName:
            strModeName = strModeName + chr(per)
        print("device model name: %s" % strModeName)

        strSerialNumber = ""
        for per in stDeviceList.DeviceInfo[i].chSerialNumber:
            strSerialNumber = strSerialNumber + chr(per)
        print("device SerialNumber: %s" % strSerialNumber)

    # 创建相机示例
    camera=Mv3dRgbd()
    # nConnectionNum = input("please input the number of the device to connect:")
    nConnectionNum = 0
    if int(nConnectionNum) >= nDeviceNum.value:
        print("intput error!")
        os.system('pause')
        sys.exit()

    # 打开设备   
    ret=camera.MV3D_RGBD_OpenDevice(pointer(stDeviceList.DeviceInfo[int(nConnectionNum)]))
    if ret!=0:
        print ("MV3D_RGBD_OpenDevice fail! ret[0x%x]" % ret)
        os.system('pause')
        sys.exit()

    # 开始取流
    ret=camera.MV3D_RGBD_Start()
    if ret != 0:
        print ("start fail! ret[0x%x]" % ret)
        camera.MV3D_RGBD_CloseDevice()
        os.system('pause')
        sys.exit()

    time_start=time.time()
    save_dir = '/f/pro/Hik/code/sensing3d/data/hik/00'
    depth_dir = os.path.join(save_dir, 'depth')
    image_dir = os.path.join(save_dir, 'image')
    pose_dir = os.path.join(save_dir, 'pose')

    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    # 清空每个目录
    clear_folder(depth_dir)
    clear_folder(image_dir)
    clear_folder(pose_dir)

    while True:
        stFrameData=MV3D_RGBD_FRAME_DATA()
        ret=camera.MV3D_RGBD_FetchFrame(pointer(stFrameData), 5000)
        if ret==0:
            for i in range(0, stFrameData.nImageCount):
                stData = stFrameData.stImageData[i]
                print("MV3D_RGBD_FetchFrame[%d]:enImageType[%d],nWidth[%d],nHeight[%d],nDataLen[%d],nFrameNum[%d],bIsRectified[%d],enStreamType[%d],enCoordinateType[%d]" % (
                    i, stData.enImageType, stData.nWidth, stData.nHeight, stData.nDataLen, stData.nFrameNum,
                    stData.bIsRectified, stData.enStreamType, stData.enCoordinateType))

                if i == 0:
                    # 获取depth
                    p_depth = string_at(stData.pData, stData.nDataLen)
                    depth_img = np.frombuffer(p_depth, dtype=np.uint16)
                    depth_img = depth_img.reshape((stData.nHeight, stData.nWidth))
                    new_width = 640
                    new_height = 480
                    depth_img = cv2.resize(depth_img, (new_width, new_height))
                    cv2.imwrite(os.path.join(save_dir, 'depth', f'{stData.nFrameNum}.png'), depth_img)

                elif i == 1:
                    # 获取image
                    pp = ctypes.string_at(stData.pData, stData.nDataLen)
                    img = np.frombuffer(pp, dtype=np.uint8)
                    img11 = img.reshape((stData.nHeight, stData.nWidth, 2))
                    image = cv2.cvtColor(img11, cv2.COLOR_YUV2BGR_YUYV)
                    new_width = 640
                    new_height = 480
                    image = cv2.resize(image, (new_width, new_height))

                    print("bgf.shape:", image.shape)
                    cv2.imwrite(os.path.join(save_dir, 'image', f'{stData.nFrameNum}.png'), image)

        else:
            print("no data[0x%x]" % ret)

        time_end=time.time()
        sum_t=time_end - time_start
        # 取流超过s后退出
        if sum_t>sample_time:
            break 

    # 停止取流
    ret=camera.MV3D_RGBD_Stop()
    if ret != 0:
        print ("stop fail! ret[0x%x]" % ret)
        os.system('pause')
        sys.exit()

    # 销毁句柄
    ret=camera.MV3D_RGBD_CloseDevice()
    if ret != 0:
        print ("CloseDevice fail! ret[0x%x]" % ret)
        os.system('pause')
        sys.exit()
    
    sys.exit()
