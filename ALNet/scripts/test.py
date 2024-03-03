from torch.utils import data
import sys
import os
import cv2

from datasets import get_dataset
from models import get_model
from loss import *
from scripts.utils import get_pose_err

import time
import matplotlib.pyplot as plt

def test(args, save_pose=False):
    if True:
        # sys.path.insert(0, '/home/dk/ghb/EAAINet/pnpransac')
        sys.path.append('/home/ghb/ALNet/pnpransac')
        from pnpransac import pnpransac
    if args.dataset == '7S':
        dataset = get_dataset('7S')
    if args.dataset == '12S':
        dataset = get_dataset('12S')
    if args.dataset == 'my':
        dataset = get_dataset('my')

    test_dataset = dataset(args.data_path, args.dataset, args.scene, split='test', model=args.model, aug='False')

    testloader = data.DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=False)
    intrinsics_color = test_dataset.intrinsics_color # 3*3 

    pose_solver = pnpransac(intrinsics_color[0,0], intrinsics_color[1,1], intrinsics_color[0,2], intrinsics_color[1,2])

    torch.set_grad_enabled(False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model, args.dataset)
    model_state = torch.load(args.resume,
                map_location=device)['model_state']
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    x = np.linspace(4, 640 - 4, 80)
    y = np.linspace(4, 480 - 4, 60)
    xx, yy = np.meshgrid(x, y)  # [60 80]
    pcoord = np.concatenate((np.expand_dims(xx, axis=2), np.expand_dims(yy, axis=2)), axis=2)
    rot_err_list = []
    transl_err_list = []

    # filename = "office.npy"
    # new_coords_list = []

    filename2 = "gt_red.npy"
    gt_list = []

    filename3 = "pre_red.npy"
    pre_list = []
    # i = 0
    # for _, (img, pose) in enumerate(testloader):
    #     i=i+1
    #     print(i)
    # exit()

    output_folder = "/mnt/share/sda1/modelghb/12s/k544uncen/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    num = -1
    for _, (img, pose) in enumerate(testloader):
        num = num + 1
        print(num)
        with torch.no_grad():
            img = img.to(device) # 1 3 480 640

            ##
            img1 = np.squeeze(img) # 3 480 640
            img1 = img1.permute(1,2,0) # 480 640 3
            img1 = img1[4::8, 4::8, :] # 60 80 3
            img1 = img1.reshape(-1,3).cpu().data.numpy()
            # print(img1.shape)
            # exit()
            #开始时间计算
            # start_time = time.time()

            coord, uncertainty = model(img)
            #结束时间计算
            # end_time = time.time()
            # elapsed_time = end_time - start_time
            # print(f"代码执行时间：{elapsed_time}秒")
            # exit()
        
        # ##将uncer存为图片
        # plt.imshow(torch.sum(uncertainty[0], dim=0).detach().cpu().numpy(),cmap="jet")
        # plt.colorbar()
        # plt.savefig(os.path.join(output_folder, f"uncer_{num}.png"))
        # # plt.show()
        # plt.close()

        #torch.Tensor'
        ##1 3 60 80
        coord = np.transpose(coord.cpu().data.numpy()[0,:,:,:], (1,2,0))
        #'numpy.ndarray'
        # 60 80 3
        # print(coord.shape)
        # print(type(coord))
        uncertainty = np.transpose(uncertainty[0].cpu().data.numpy(), (1,2,0))
        # 60 80 1
        coord = np.concatenate([coord,uncertainty],axis=2)
        # 60 80 4
        coord = np.ascontiguousarray(coord)
        # 60 80 4 内存连续
        pcoord = np.ascontiguousarray(pcoord)
        pcoord = pcoord.reshape(-1,2)
        coords = coord[:,:,0:3].reshape(-1,3)
        # 4800 3
        confidences = coord[:,:,3].flatten().tolist()
        ###
        # new_coords = coord.reshape(-1,4) #4800 4 'numpy.ndarray'
        # new_coords = np.concatenate([new_coords,img1],axis=1) #x y z uncer r g b
        # print(new_coords.shape)
        # exit()
        # try:
        #     # 尝试加载已有数据
        #     existing_data = np.load(filename)
        #     # 合并新数据与已有数据
        #     combined_data = np.vstack((existing_data, new_coords))
        # except FileNotFoundError:
        #     # 如果文件不存在，创建一个新数组并保存新数据
        #     combined_data = new_coords
        # # 保存新的数据到文件
        # np.save(filename, combined_data)
        # new_coords_list.append(new_coords)
        #print('ooooook')
        ###
        # start_time = time.time()

        coords_filtered = []
        coords_filtered_2D = []
        for i in range(len(confidences)):
            # if confidences[i] < 0.2:
            #     coords_filtered.append(coords[i])
            #     coords_filtered_2D.append(pcoord[i])
            if confidences[i] > 0.0:
                coords_filtered.append(coords[i])
                coords_filtered_2D.append(pcoord[i])

        coords_filtered = np.vstack(coords_filtered)
        coords_filtered_2D = np.vstack(coords_filtered_2D)

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"代码执行时间：{elapsed_time}秒")
        # exit()

        rot, transl = pose_solver.RANSAC_loop(coords_filtered_2D.astype(np.float64), coords_filtered.astype(np.float64), 256)
        
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"代码执行时间：{elapsed_time}秒")
        # exit()
        pose_gt = pose.data.numpy()[0,:,:]  # [4 4]
        pose_est = np.eye(4)        # [4 4]
        pose_est[0:3,0:3] = cv2.Rodrigues(rot)[0].T             # Rwc
        pose_est[0:3,3] = -np.dot(pose_est[0:3,0:3], transl)    # twc

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"代码执行时间：{elapsed_time}秒")
        # exit()

        #dushuju
        gtdata = pose_gt[0:3,3]
        gt_list.append(gtdata)

        predata = pose_est[0:3,3]
        pre_list.append(predata)
        #
        # print(pose_gt.shape)
        # print(pose_est.shape)
        # exit()
        transl_err, rot_err = get_pose_err(pose_gt, pose_est)
        rot_err_list.append(rot_err)
        transl_err_list.append(transl_err)
        print('step:{}, Pose error: {}m, {}\u00b0，changdu:{}'.format(_ ,transl_err, rot_err,len(coords_filtered_2D)))
    
    ###
    # my_array = np.array(new_coords_list)
    # np.save(filename, my_array)
    ###
    # gt_array = np.array(gt_list)
    # print(gt_array.shape)
    # np.save(filename2, gt_array)
    # ##
    # pre_array = np.array(pre_list)
    # print(pre_array.shape)
    # np.save(filename3, pre_array)
    # ##
    # exit()

    results = np.array([transl_err_list, rot_err_list]).T   # N 2
    np.savetxt(os.path.join(args.output,
            'pose_err_{}_{}_{}_coord.txt'.format(args.dataset,
            args.scene.replace('/','.'), args.model)), results)
    print('Accuracy: {}%'.format(np.sum((results[:,0] <= 0.050)
                * (results[:,1] <= 5)) * 1. / len(results) * 100))
    print('Median pose error: {}m, {}\u00b0'.format(np.median(results[:,0]),
            np.median(results[:,1])))
    print('Average pose error: {}m, {}\u00b0'.format(np.mean(results[:,0]),
            np.mean(results[:,1])))
    print('stddev: {}m, {}\u00b0'.format(np.std(results[:,0],ddof=1),
            np.std(results[:,1],ddof=1)))








































