import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

data_config = "config.yaml"

if __name__ == '__main__':
    model = YOLO('D:/python123/ultralytics/runs/train/exp18/weights/best.pt')
    # 如何切换模型版本, 上面的ymal文件可以改为 yolov11s.yaml就是使用的v11s,
    # 类似某个改进的yaml文件名称为yolov11-XXX.yaml那么如果想使用其它版本就把上面的名称改为yolov11l-XXX.yaml即可（改的是上面YOLO中间的名字不是配置文件的）！
    model.load('D:/python123/ultralytics/runs/train/exp18/weights/best.pt') # 是否加载预训练权重,科研不建议大家加载否则很难提升精度
    # results = model.val()  # 运行验证
    #
    # # 输出评估结果
    # print("验证集结果：")
    # print(f"Box(P): {results['metrics']['box']['P']:.3f}")
    # print(f"R: {results['metrics']['box']['R']:.3f}")
    # print(f"mAP50: {results['metrics']['mAP50']:.3f}")
    # print(f"mAP50-95: {results['metrics']['mAP50-95']:.3f}")

    model.train(data=data_config,
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                cache=False,
                imgsz=640,
                epochs=2002,
                single_cls=False,  # 是否是单类别检测
                batch=8,
                close_mosaic=0,
                workers=0,
                device='0',
                optimizer='SGD',  # using SGD 优化器 默认为auto建议大家使用固定的.
                # resume=True, # 续训的话这里填写True, yaml文件的地方改为lats.pt的地址,需要注意的是如果你设置训练200轮次模型训练了200轮次是没有办法进行续训的.
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                project='runs/train',
                name='exp',
                )
