Data Loading and Processing Tutorial：
本文讲述了数据读取和预处理。
两种格式：
1、有文本标注csv格式如下，图片路径，标注点坐标1，标注点坐标2，标注点坐标3，标注点坐标4...........：
csv数据格式：
image_name,part_0_x,part_0_y,part_1_x,part_1_y,part_2_x, ... ,part_67_x,part_67_y
0805personali01.jpg,27,83,27,98, ... 84,134
1084239450_e76e00b7e7.jpg,70,236,71,257, ... ,128,312

2、采用文件夹作为lable标签格式如下：
假定图像数据集如下：
root/ants/xxx.png
root/ants/xxy.jpeg
root/ants/xxz.png
.
.
.
root/bees/123.jpg
root/bees/nsdf3.png
root/bees/asd932_.png


第一种：
scikit-image: 用于图像输入输出以及transforms
pandas: 容易对csv格式进行解析

数据下载 https://download.pytorch.org/tutorial/faces.zip
data/faces/face_landmarks.csv 68个点
该文件中包含了create_landmark_dataset.py（用来对图像进行标注写入）和标注文件face_landmarks.csv、以及数据图像

csv数据格式：
image_name,part_0_x,part_0_y,part_1_x,part_1_y,part_2_x, ... ,part_67_x,part_67_y
0805personali01.jpg,27,83,27,98, ... 84,134
1084239450_e76e00b7e7.jpg,70,236,71,257, ... ,128,312


#图像大小不一致。需要使用预处理代码来使得图片的大小一致。 利用transforms
transforms使用：
tsfm = Transform(params)
transformed_sample = tsfm(sample)
"""
Rescale: to scale the image
RandomCrop: to crop from image randomly. This is data augmentation.
ToTensor: to convert the numpy images to torch images (we need to swap axes)
"""

#使用简单的for循环来迭代数据，我们将会失去很多的特性，特别是：
#batching the data                              批处理数据
#shuffling the data                             整理数据
#load the data in parallel using multiprocessing workers .使用多线程读取数据
#使用torch.utils.data.DataLoader可以提供上面那些特性。参数明显，一个有趣的参数就是collate_fn.您可以使用collate_fn来具体的指定样本batch
#一般默认的就可以
transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                           root_dir='data/faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)
                        
第二种数据图片读取的格式：ImageFolder
假定图像数据集如下：
root/ants/xxx.png
root/ants/xxy.jpeg
root/ants/xxz.png
.
.
.
root/bees/123.jpg
root/bees/nsdf3.png
root/bees/asd932_.png
在这里ants,bees是标签， 可以RandomHorizontalFlip, Scale来对数据进行转换，那么我们数据读取器可以写成这样子：
import torch
from torchvision import transforms, datasets

data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
                                           transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)
                        
                        
