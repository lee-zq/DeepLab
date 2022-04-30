import numpy as np
import onnxruntime as ort
import argparse, os
from lib import CIFARDataset

def onnxruntime_test(session, testloader):
    print("Start Testing!")
    input_name = session.get_inputs()[0].name
    correct = 0
    total = 0   # 计数归零（初始化）
    for data in testloader:
        images, labels = data
        images, labels = images.numpy(), labels.numpy()
        outputs = session.run(None, {input_name:images})
        predicted = np.argmax(outputs[0], axis=1)  # 取得分最高的那个类
        total += labels.shape[0]                        # 累加样本总数
        correct += (predicted == labels).sum()        # 累加预测正确的样本个数
    acc = correct / total
    print('ONNXRuntime Test Acc is: %.2f%%' % (100*acc))
            
if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser("CNN backbone on cifar10")
    parser.add_argument('--onnx', default='./output/test_densenet/densenet_best.onnx')
    args = parser.parse_args()

    NUM_CLASS =10
    BATCH_SIZE = 128  # 批处理尺寸(batch_size)

    # 数据集迭代器
    data_path="./data"
    dataset = CIFARDataset(dataset_path=data_path, batchsize=BATCH_SIZE)
    _, testloader = dataset.get_cifar10_dataloader()

    # 构建session
    sess = ort.InferenceSession(args.onnx, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    #onnxruntime推理
    onnxruntime_test(sess, testloader)