import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


set = "V202"
ver = "2"
if set == "MH01":
    offset = 20
if set == "MH02":
    offset = 20
if set == "MH03":
    offset = 49
if set == "MH04":
    offset = 35
if set == "MH05":
    offset = 31
if set == "V101":
    offset = 23
if set == "V102":
    offset = 22
if set == "V103":
    offset = 38
if set == "V201":
    offset = 27
if set == "V202":
    offset = 27
if set == "V203":
    offset = 26

def read_txt_groundtruth(namefile):
    timestamp = list()
    pos_x = list()
    pos_y = list()
    pos_z = list()
    quat_w = list()
    quat_x = list()
    quat_y = list()
    quat_z = list()

    with open(namefile) as f:
        lines = f.readlines()
        i = 0
        for row_ in lines:
            if (i > 0):
                row = row_.split(',')
                timestamp.append(float(row[0])/1e9)
                pos_x.append(float(row[1]))
                pos_y.append(float(row[2]))
                pos_z.append(float(row[3]))
                quat_w.append(float(row[4]))
                quat_x.append(float(row[5]))
                quat_y.append(float(row[6]))
                quat_z.append(float(row[7]))
            i += 1
        
    np_timestamp = np.array(timestamp)
    np_pos = np.array([pos_x, pos_y, pos_z])
    np_quat = np.array([quat_x, quat_y, quat_z, quat_w])

    return np_timestamp, np_pos, np_quat


def read_txt_result(namefile):
    timestamp = list()
    pos_x = list()
    pos_y = list()
    pos_z = list()
    quat_w = list()
    quat_x = list()
    quat_y = list()
    quat_z = list()

    with open(namefile) as f:
        i = 0
        lines = f.readlines()
        for row_ in lines:
            if (i >= offset):
                row = row_.split()
                timestamp.append(float(row[0]))
                pos_x.append(float(row[1]))
                pos_y.append(float(row[2]))
                pos_z.append(float(row[3]))
                quat_x.append(float(row[4]))
                quat_y.append(float(row[5]))
                quat_z.append(float(row[6]))
                quat_w.append(float(row[7]))
            i += 1
        
    np_timestamp = np.array(timestamp)
    np_pos = np.array([pos_x, pos_y, pos_z])
    np_quat = np.array([quat_x, quat_y, quat_z, quat_w])

    return np_timestamp, np_pos, np_quat



time_GT, pos_GT, quat_GT = read_txt_groundtruth("../ORB_SLAM3/evaluation/Ground_truth/EuRoC_left_cam/"+set+"_GT.txt")
time_sim, pos_sim, quat_sim = read_txt_result("result/CameraTrajectory_"+set+"_"+ver+".txt")
#time_sim, pos_sim, quat_sim = read_txt_result("CameraTrajectory.txt")

base_rot = R.from_quat((-quat_GT[0,0], -quat_GT[1,0], -quat_GT[2,0], quat_GT[3,0])).as_matrix()

print(base_rot)
print("SIZE GT = ", pos_GT[:,0])
print("SIZE sim = {}".format(pos_sim[:,0]))

pose_list = list()
quat_list = list()
print(pos_sim.shape)
for i in range(pos_sim.shape[1]):
    new_cam = base_rot@pos_sim[:,i]
    new_cam[0] += pos_GT[0,0]
    new_cam[1] += pos_GT[1,0]
    new_cam[2] += pos_GT[2,0] + 0.15
    pose_list.append((new_cam[0], new_cam[1], new_cam[2]))

for i in range(quat_sim.shape[1]):
    rot_temp = R.from_quat((quat_sim[0,i], quat_sim[1,i], quat_sim[2,i], quat_sim[3,i])).as_matrix()

    new_rot = rot_temp@base_rot
    new_quat = R.as_quat(R.from_matrix(new_rot))
    quat_list.append((new_quat[0], new_quat[1], new_quat[2], new_quat[3]))

cam_pose = np.array(pose_list)
cam_quat = np.array(quat_list)

print(cam_pose.shape)


name = ['Position X', 'Position Y', 'Position Z']

for i in range(3):
    plt.figure(i)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.plot(time_GT-time_GT[0], pos_GT[i,:], linewidth=2, label='Ground truth')
    plt.plot(time_sim[:]-time_sim[0], cam_pose[:,i], linewidth=2, label='SLAM estimate')
    plt.xlabel('Time [s]',{'size':15})
    plt.ylabel('Position [m]',{'size':15})
    plt.grid(True)
    plt.title(name[i]+ " " + set,{'size':15})
    plt.legend(["Ground truth", "SLAM estimate"],fontsize=15)
    plt.subplots_adjust(top=0.925, bottom=0.11, left=0.055, right=0.955, hspace=0.2, wspace=0.2)
 
'''
name = ['Quaternion X', 'Quaternion Y', 'Quaternion Z', 'Quaternion W']
for i in range(4):
    plt.figure(i+3)
    plt.plot(time_GT-time_GT[0], quat_GT[i,:])
    plt.plot(time_sim[offset:]-time_sim[offset], cam_quat[offset:,i])
    plt.xlabel('Time [s]')
    plt.ylabel('Quaternion')
    plt.grid(True)
    plt.title(name[i]+ " " + set)
    plt.legend(["Ground Truth", "SLAM estimate"])
'''
    
#ax = plt.figure(10).add_subplot(projection='3d')

#ax.plot(pos_GT[0,:], pos_GT[1,:], pos_GT[2,:], label='Ground Truth')
#ax.plot(cam_pose[offset:,0], cam_pose[offset:,1], cam_pose[offset:,2], label='SLAM estimate')

T = min([pos_GT.shape[1], cam_pose.shape[0]])

error = np.zeros((T,1))
error_sum = 0
error_tot = 0

for i in range(T):
    error[i] = np.linalg.norm(pos_GT[:,i]-cam_pose[i,:])
    error_sum += error[i]
    error_tot += error[i]*error[i]

rmse = error_sum/T
std_dev = np.sqrt(T*error_tot - error_sum**2)/T

plt.figure()
plt.plot(time_GT[:T]-time_GT[0], error, color='black', linewidth=2,label='error')
plt.plot(time_GT[:T]-time_GT[0], rmse*np.ones(error.shape), linestyle='--', color='black', linewidth=3, label='error mean')
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Time [s]',{'size':15})
plt.ylabel('Error [m]',{'size':15})
plt.title('Absolute error on the dataset '+set,{'size':15})
plt.legend(['Prediction error', 'Mean error'],fontsize=15)
plt.subplots_adjust(top=0.925, bottom=0.11, left=0.055, right=0.955, hspace=0.2, wspace=0.2)

print("MEAN ABS ERROR = ", rmse)
print("STD DEV = ", std_dev)



###### OFFLINE #########################àà
time_sim, pos_sim, quat_sim = read_txt_result("result/OfflineCameraTrajectory_"+set+"_"+ver+".txt")

base_rot = R.from_quat((-quat_GT[0,0], -quat_GT[1,0], -quat_GT[2,0], quat_GT[3,0])).as_matrix()

print(base_rot)
print("SIZE GT = ", pos_GT[:,0])
print("SIZE sim = {}".format(pos_sim[:,0]))

pose_list = list()
quat_list = list()
print(pos_sim.shape)
for i in range(pos_sim.shape[1]):
    new_cam = base_rot@pos_sim[:,i]
    new_cam[0] += pos_GT[0,0]
    new_cam[1] += pos_GT[1,0]
    new_cam[2] += pos_GT[2,0] + 0.15
    pose_list.append((new_cam[0], new_cam[1], new_cam[2]))

for i in range(quat_sim.shape[1]):
    rot_temp = R.from_quat((quat_sim[0,i], quat_sim[1,i], quat_sim[2,i], quat_sim[3,i])).as_matrix()

    new_rot = rot_temp@base_rot
    new_quat = R.as_quat(R.from_matrix(new_rot))
    quat_list.append((new_quat[0], new_quat[1], new_quat[2], new_quat[3]))

cam_pose = np.array(pose_list)
cam_quat = np.array(quat_list)
    
#ax = plt.figure(10).add_subplot(projection='3d')

#ax.plot(pos_GT[0,:], pos_GT[1,:], pos_GT[2,:], label='Ground Truth')
#ax.plot(cam_pose[offset:,0], cam_pose[offset:,1], cam_pose[offset:,2], label='SLAM estimate')

T = min([pos_GT.shape[1], cam_pose.shape[0]])

error = np.zeros((T,1))
error_sum = 0
error_tot = 0

for i in range(T):
    error[i] = np.linalg.norm(pos_GT[:,i]-cam_pose[i,:])
    error_sum += error[i]
    error_tot += error[i]*error[i]

rmse = error_sum/T
std_dev = np.sqrt(T*error_tot - error_sum**2)/T

plt.figure()
plt.plot(time_GT[:T]-time_GT[0], error, color='black', linewidth=1,label='error')
plt.plot(time_GT[:T]-time_GT[0], rmse*np.ones(error.shape), linestyle='--', color='black', linewidth=3, label='error mean')
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Time [s]',{'size':15})
plt.ylabel('Error [m]',{'size':15})
plt.title('Absolute error (Offline) on the dataset '+set,{'size':15})
plt.legend(['Prediction error', 'Mean error'],fontsize=15)
plt.subplots_adjust(top=0.925, bottom=0.11, left=0.055, right=0.955, hspace=0.2, wspace=0.2)

print("MEAN ABS ERROR = ", rmse)
print("STD DEV = ", std_dev)

plt.show()