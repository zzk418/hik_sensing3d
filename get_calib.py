# -- coding: utf-8 --
import json
import pprint
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

def extract_calib_info(calib_info):
    # 提取内参矩阵
    intrinsic_matrix = [calib_info.stIntrinsic.fData[i] for i in range(9)]

    # 提取畸变系数
    distortion_coefficients = [calib_info.stDistortion.fData[i] for i in range(12)]

    return {
        'intrinsic_matrix': intrinsic_matrix,
        'distortion_coefficients': distortion_coefficients,
    }

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
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

    # 获取 depth 相机参数
    depth_calib = MV3D_RGBD_CALIB_INFO()
    camera.MV3D_RGBD_GetCalibInfo(1, pointer(depth_calib))
    
    depth_calib_dict = extract_calib_info(depth_calib)
    print("depth_calib_dict:")
    save_to_json(depth_calib_dict, "depth_calib.json")
    pprint.pprint(depth_calib_dict)

    # 获取 rgb 相机参数
    rgb_calib = MV3D_RGBD_CALIB_INFO()
    camera.MV3D_RGBD_GetCalibInfo(2, pointer(rgb_calib))
    
    rgb_calib_dict = extract_calib_info(rgb_calib)
    save_to_json(rgb_calib_dict, "rgb_calib.json")
    print("rgb_calib_dict:")
    pprint.pprint(rgb_calib_dict)