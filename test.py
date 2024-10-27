# import warnings
# import os
# from ultralytics import YOLO
# from PIL import Image
# import numpy as np
#
# # 忽略警告
# warnings.filterwarnings('ignore')
#
# # 加载训练好的模型（根据您的模型保存路径进行调整）
# model = YOLO('D:/python123/ultralytics/best.pt')  # 根据模型保存位置进行调整
#
# # 定义包含图像的文件夹
# input_folder = r'H:\broken'
# output_folder = r'H:\broken_识别结果'
#
# # 确保输出文件夹存在
# os.makedirs(output_folder, exist_ok=True)
#
# # 支持的图像格式
# image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
#
# # 用于保存处理后的图像
# processed_images = []
# combined_counter = 0  # 计数器，保存拼接图像的编号
#
# # 遍历输入文件夹中的所有文件
# for filename in os.listdir(input_folder):
#     # 检查文件是否为图像
#     if any(filename.lower().endswith(ext) for ext in image_extensions):
#         # 加载图像
#         img_path = os.path.join(input_folder, filename)
#         img = Image.open(img_path)
#
#         # 对图像进行检测
#         results = model(img)
#
#         # 获取标注后的图像，假设结果中第一项就是输出图像
#         annotated_image = results[0].plot()  # 返回的是一个 NumPy 数组
#
#         # 将 NumPy 数组转换为 PIL 图像
#         annotated_image = Image.fromarray(annotated_image)
#
#         processed_images.append(annotated_image)
#
#         print(f"处理并保存: {filename}")
#
#         # 每4张图像拼接一次
#         if len(processed_images) == 4:
#             # 拼接图像
#             combined_image = Image.new('RGB', (annotated_image.width * 2, annotated_image.height * 2))
#
#             for index, image in enumerate(processed_images):
#                 x_offset = (index % 2) * annotated_image.width
#                 y_offset = (index // 2) * annotated_image.height
#                 combined_image.paste(image, (x_offset, y_offset))
#
#             # 保存拼接后的图像，使用计数器确保唯一性
#             combined_output_path = os.path.join(output_folder, f"combined_{combined_counter}.jpg")
#             combined_image.save(combined_output_path)
#             print(f"拼接并保存: {combined_output_path}")
#             combined_counter += 1  # 增加计数器
#             # 清空列表以便下一次拼接
#             processed_images = []
#
# # 处理完剩余未拼接的图像
# if processed_images:
#     combined_image = Image.new('RGB', (annotated_image.width * 2, annotated_image.height * 2))
#
#     for index, image in enumerate(processed_images):
#         x_offset = (index % 2) * annotated_image.width
#         y_offset = (index // 2) * annotated_image.height
#         combined_image.paste(image, (x_offset, y_offset))
#
#     combined_output_path = os.path.join(output_folder, f"combined_{combined_counter}.jpg")
#     combined_image.save(combined_output_path)
#     print(f"拼接并保存: {combined_output_path}")
#
# print("所有图像已处理并保存。")
# import os
#
# # 指定包含 TXT 文件的文件夹路径
# folder_path = r'H:\out_broken'
#
# # 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     if filename.endswith('.txt'):
#         txt_file_path = os.path.join(folder_path, filename)
#
#         # 读取当前 TXT 文件
#         with open(txt_file_path, 'r') as file:
#             lines = file.readlines()
#
#         # 处理每一行
#         modified_lines = []
#         for line in lines:
#             parts = line.split()  # 按空格分割
#             if parts:  # 确保行不为空
#                 # 将第一个数字减少 15
#                 if int(parts[0])>14:
#                     class_id = str(int(parts[0])-15)
#                 else:
#                     class_id = parts[0]
#                 # 生成新的行
#                 new_line = f"{class_id} " + " ".join(parts[1:]) + "\n"
#                 modified_lines.append(new_line)
#
#         # 将修改后的内容写回原文件
#         with open(txt_file_path, 'w') as file:
#             file.writelines(modified_lines)
#
#         print(f"已处理文件: {filename}")
#
# print("所有文件已处理完毕。")
import os


def process_label_files(directory):
    # 遍历目标目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)

            # 读取文件内容并处理
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # 处理每一行
            processed_lines = []
            for line in lines:
                # 按空格分割行中的每个元素
                parts = line.split()
                if parts:
                    # 去掉小数点并转换为整数
                    label = parts[0].split('.')[0]  # 去掉数字的后面的部分
                    processed_line = f"{label} " + " ".join(parts[1:])  # 保留其他内容与空格
                    processed_lines.append(processed_line)

            # 写入到文件
            with open(file_path, 'w') as file:
                file.write("\n".join(processed_lines))


# 指定要处理的目录
directory_path = 'D:\python123/ultralytics\datas\label/train'  # 替换为您的目录路径
process_label_files(directory_path)
