import cv2
import numpy as np
import os

# 定义数据增强函数
def data_augmentation(image):
    # 旋转
    angle = np.random.randint(-15, 15)  # 随机选择旋转角度
    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # 剪裁
    crop_size = np.random.randint(0, 10)  # 随机选择剪裁尺寸
    image = image[crop_size:image.shape[0] - crop_size, crop_size:image.shape[1] - crop_size]


    # 调整亮度
    brightness = np.random.randint(-30, 30)  # 随机选择亮度调整值
    image = cv2.convertScaleAbs(image, alpha=1.0, beta=brightness)

    return image

# 定义数据增强文件夹路径
train_data_dir = "D:\\桌面\\data\\data"
augmented_data_dir = "D:\\桌面\\data\\dataaugmentation"



# 遍历训练数据文件夹中的图像文件
for filename in os.listdir(train_data_dir):
    if filename.endswith('.jpg'):  # 假设图像文件格式为.jpg
        image_path = os.path.join(train_data_dir, filename)
        image = cv2.imread(image_path)

        # 扩充数据
        for i in range(5):  # 每张图像扩充5次
            augmented_image = data_augmentation(image)

            # 保存扩充后的图像
            augmented_filename = filename.split('.')[0] + '_augmented_' + str(i) + '.jpg'
            augmented_image_path = os.path.join(augmented_data_dir, augmented_filename)
            cv2.imwrite(augmented_image_path, augmented_image)


