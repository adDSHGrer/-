import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os


# 数据增强函数
def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),  # 水平翻转
        A.RandomBrightnessContrast(p=0.5),  # 随机亮度和对比度
        A.GaussNoise(var_limit=(10, 50), p=0.3),  # 添加高斯噪声，方差范围为10到50
        A.Normalize(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    ))


def load_image_and_labels(image_path, label_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with open(label_path, 'r') as f:
        labels = []
        bboxes = []
        for line in f.readlines():
            label_info = line.strip().split()
            class_id = int(label_info[0])
            bbox = list(map(float, label_info[1:]))
            labels.append(class_id)
            bboxes.append(bbox)
    return image, bboxes, labels


def denormalize_image(image_tensor):
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).clip(0, 255).astype('uint8')
    return image


def save_augmented_image_and_labels(image, bboxes, class_labels, img_save_dir, label_save_dir, image_name):
    os.makedirs(save_dir, exist_ok=True)
    augmented_image_path = os.path.join(img_save_dir, f'{image_name}.jpg')
    augmented_label_path = os.path.join(label_save_dir, f'{image_name}.txt')

    # 保存图像
    cv2.imwrite(augmented_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # 保存标签
    with open(augmented_label_path, 'w') as f:
        for bbox, class_id in zip(bboxes, class_labels):
            bbox_str = ' '.join(map(str, bbox))
            f.write(f'{class_id} {bbox_str}\n')


def augment_and_split_data(train_images_dir, train_labels_dir, save_dir, transform, augment_times=10, split_ratio=0.6):
    image_files = os.listdir(train_images_dir)

    image_train_dir = os.path.join(save_dir, 'image', 'train')
    image_val_dir = os.path.join(save_dir, 'image', 'val')
    label_train_dir = os.path.join(save_dir, 'label', 'train')
    label_val_dir = os.path.join(save_dir, 'label', 'val')

    os.makedirs(image_train_dir, exist_ok=True)
    os.makedirs(image_val_dir, exist_ok=True)
    os.makedirs(label_train_dir, exist_ok=True)
    os.makedirs(label_val_dir, exist_ok=True)

    for image_file in image_files:
        image_path = os.path.join(train_images_dir, image_file)
        label_path = os.path.join(train_labels_dir, image_file.replace('.jpg', '.txt'))

        image, bboxes, class_labels = load_image_and_labels(image_path, label_path)
        if image is None or bboxes is None or class_labels is None:
            continue

        augmented_images = []
        augmented_bboxes_list = []
        augmented_class_labels_list = []

        for i in range(augment_times):
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            augmented_image = augmented['image']
            augmented_bboxes = augmented['bboxes']
            augmented_class_labels = augmented['class_labels']

            # 检查是否有有效边界框
            if augmented_bboxes:  # 只保存有效的增强图像
                augmented_images.append(augmented_image)
                augmented_bboxes_list.append(augmented_bboxes)
                augmented_class_labels_list.append(augmented_class_labels)

        total_images = len(augmented_images)
        split_index = int(total_images * split_ratio)

        for idx, (augmented_image, augmented_bboxes, augmented_class_labels) in enumerate(
                zip(augmented_images, augmented_bboxes_list, augmented_class_labels_list)):
            denormalized_image = denormalize_image(augmented_image)

            if idx < split_index:
                save_augmented_image_and_labels(denormalized_image, augmented_bboxes,
                                                augmented_class_labels, image_train_dir, label_train_dir,
                                                f"{image_file.split('.')[0]}_hrg10_{idx}")
            else:
                save_augmented_image_and_labels(denormalized_image, augmented_bboxes,
                                                augmented_class_labels, image_val_dir, label_val_dir,
                                                f"{image_file.split('.')[0]}_hrg10_{idx}")


# 设置路径
train_images_dir = 'H:/broken'  # 替换为你的训练图像路径
train_labels_dir = 'H:/out_broken'  # 替换为你的训练标签路径
save_dir = 'D:/python123/ultralytics/datas'  # 替换为你想保存增强后数据的路径

train_transforms = get_train_transforms()
augment_and_split_data(train_images_dir, train_labels_dir, save_dir, train_transforms)
