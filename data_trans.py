import h5py
import pickle
import torch
import numpy as np

#读取.h5文件 
filename = '/root/autodl-tmp/MMRNS-Datasets/emb_data/MMKB_img_BEIT_16-224.h5'
save_filename = '/root/autodl-tmp/MMRNS-Datasets/emb_data/DB15K-visual.pth'
# f = h5py.File(filename, 'r')
# vgg_feats = f["m.010016"] [:]
# print(vgg_feats)
all_tensors = {}
with h5py.File(filename, 'r') as file:
    # 打印文件中的所有组和数据集
    for key in file.keys():
        # # print(key)
        feats_h5 = file[key] 
        # print(vgg_feats_h5)    
        # 将HDF5数据转换为NumPy数组
        feats_np = np.array(feats_h5)
            
        # 将NumPy数组转换为PyTorch张量
        feats_tensor = torch.from_numpy(feats_np)
            
        # 将张量保存到字典中
        all_tensors[key]=feats_tensor.numpy()
        
#         vgg_feats_h5 = file[key] 
#         # 将HDF5数据转换为NumPy数组
#         vgg_feats_np = np.array(vgg_feats_h5)
            
#         # 将NumPy数组转换为PyTorch张量，并添加一个批次维度
#         vgg_feats_tensor = torch.from_numpy(vgg_feats_np).unsqueeze(0)
            
        # 将张量保存到列表中
        # all_tensors.append(vgg_feats_tensor)
        # print(all_tensors)
        # break

# # 将列表中的所有张量堆叠成一个二维张量
# all_tensors_stacked = torch.cat(all_tensors, dim=0)

# 保存整个字典到.pth文件
torch.save(all_tensors, save_filename)
with open(save_filename, 'wb') as f:
    pickle.dump(all_tensors, f)
print("Done!")
# print(text_features["0"].shape)
# print(len(text_features))


# #.h5中文件转存为pkl（字典）
# filenames_file = '/home/1718/mhh/IMF-Pytorch-main/dataset/DB15K/DB15K_ImageIndex.txt'
# hdf5_file = '/home/1718/mhh/IMF-Pytorch-main/dataset/DB15K/DB15K_ImageData.h5'
# save_filename = '/home/1718/mhh/IMF-Pytorch-main/dataset/DB15K/img_features.pkl'

# all_tensors = {}

# # 打开文件并逐行读取
# with open(filenames_file, 'r') as f:
#     for line in f:
#         # 移除字符串末尾的换行符并按空格分割字符串
#         parts = line.strip().split(' ')
#         # 假设ID是每行的最后一个元素，前面有一个空格
#         filename = parts[-1]
#         url = parts[0]  # URL是每行的第一个元素
#         filename = str(filename[-10:]) # ID是每行的最后一个元素
#         # 打开HDF5文件
#         f = h5py.File(hdf5_file, 'r')
#         vgg_feats_h5 = f[str(filename[-10:])] 
#         # print(vgg_feats_h5)    
#         # 将HDF5数据转换为NumPy数组
#         vgg_feats_np = np.array(vgg_feats_h5)
            
#         # 将NumPy数组转换为PyTorch张量
#         vgg_feats_tensor = torch.from_numpy(vgg_feats_np)
            
#         # 将张量保存到字典中
#         all_tensors[filename]=vgg_feats_tensor.numpy().squeeze()


# # 保存整个字典到.pth文件
# # torch.save(all_tensors, save_filename)
# with open(save_filename, 'wb') as f:
#     pickle.dump(all_tensors, f)
# print("Done!")



#.h5中文件转存为pkl（二维张量）
# import h5py
# import pickle
# import torch
# import numpy as np

# filenames_file = '/home/1718/mhh/IMF-Pytorch-main/dataset/DB15K/DB15K_ImageIndex.txt'
# hdf5_file = '/home/1718/mhh/IMF-Pytorch-main/dataset/DB15K/DB15K_ImageData.h5'
# save_filename = '/home/1718/mhh/IMF-Pytorch-main/dataset/DB15K/img_features.pkl'

# # 初始化一个空列表来保存二维张量
# all_tensors = []

# # 打开文件并逐行读取
# with open(filenames_file, 'r') as f:
#     for line in f:
#         # 移除字符串末尾的换行符并按空格分割字符串
#         parts = line.strip().split(' ')
#         # 假设ID是每行的最后一个元素，前面有一个空格
#         filename = parts[-1]
#         url = parts[0]  # URL是每行的第一个元素
#         filename = str(filename[-10:])  # ID是每行的最后一个元素
        
#         # 打开HDF5文件
#         with h5py.File(hdf5_file, 'r') as hf:
#             vgg_feats_h5 = hf[filename]  # 获取对应的特征数据
            
#             # 将HDF5数据转换为NumPy数组
#             vgg_feats_np = np.array(vgg_feats_h5)
            
#             # 将NumPy数组转换为PyTorch张量，并添加一个批次维度
#             vgg_feats_tensor = torch.from_numpy(vgg_feats_np).unsqueeze(0)
            
#             # 将张量保存到列表中
#             all_tensors.append(vgg_feats_tensor)
#             print(all_tensors)
#             break

# # 将列表中的所有张量堆叠成一个二维张量
# all_tensors_stacked = torch.cat(all_tensors, dim=0)

# # 保存整个二维张量到.pth文件
# torch.save(all_tensors_stacked, save_filename)
# print("Done!")



# data_load查看数据
# datasets='FB15K'
# path = '/home/1718/mhh/IMF-Pytorch-main/dataset/'+datasets+'/'
# img_features = pickle.load(open(path+'img_features.pkl', 'rb'))
# text_features = pickle.load(open(path+'text_features.pkl', 'rb'))
# 输出图像特征和文本特征的形状
# print("图像特征的形状:", img_features.shape)
# print("文本特征的形状:", text_features.shape)

# # 如果需要查看特征数据的具体内容，可以使用以下代码
# # 这里仅打印前10个图像特征和文本特征作为示例
# print("前10个图像特征:")
# print(img_features[:10])
# print("前10个文本特征:")
# print(text_features[:10])

# print(text_features["0"].shape)
# print(len(text_features))
