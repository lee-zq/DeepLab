from tqdm import tqdm
import torch, sys
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import argparse, os, time
from . import models

from lib.utils.common import Logger, print_network
from lib import CIFARDataset


def test(net, testloader, device):
    print("Start Testing!")
    net.eval()  # 将net置为评估模式（反向传播=False）
    with torch.no_grad():  # 将pytorch置为不计算梯度模式
        correct = 0
        total = 0   # 计数归零（初始化）
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):  # 加载测试集
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # 测试数据导入GPU
            outputs = net(images)  # 前向传播
            _, predicted = torch.max(outputs.data, dim=1)  # 取得分最高的那个类 (outputs.data的索引号)
            total += labels.size(0)                        # 累加样本总数
            correct += (predicted == labels).sum().item()        # 累加预测正确的样本个数
        acc = correct / total
    print('Test Acc is: %.2f%%' % (100*acc))

def export_onnx(net, testloader, output_file):
    net.eval()
    with torch.no_grad():
        for data in testloader: 
            images, labels = data

            torch.onnx.export(net, 
                            (images), 
                            output_file,
                            training=False,
                            do_constant_folding=True,
                            input_names=["img"], 
                            output_names=["output"],
                  dynamic_axes={"img": {0: "b"},"output": {0: "b"}}
                  )
            print("onnx export done!")
            break
            


def main(args):
    # 命令行参数解析
    # parser = argparse.ArgumentParser("CNN backbone on cifar10")
    # parser.add_argument('--checkpoint', default='./output/test_resnet18_10_autoaug/net_best.pth') 
    # args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # 数据集迭代器
    data_path=args.test_data_path
    dataset = CIFARDataset(dataset_path=data_path, batchsize=1)
    _, testloader = dataset.get_cifar10_dataloader()

    # 构建模型
    # net = models.LeNet(num_classes=NUM_CLASS)
    # net = models.Octresnet50(num_classes=NUM_CLASS)
    # net = models.OctNet(num_classes)
    # net = models.ghost_net()
    net = models.DenseNet(num_classes=args.batch_size)
    # net = models.DeformLeNet()
    # net = models.ResNet18()
    print_network(net)
    net.load_state_dict(state_dict=torch.load(args.outf + "/net_best.pth"))
    net.to(device)  # 转移到GPU
    
    #测试推理
    start = time.time()
    test(net, testloader, device)
    end = time.time()
    print(f"Average response time cost: {1000*(end-start)/len(testloader.dataset)} ms")

    #导出onnx模型
    output_file = args.outf + "/net_best.onnx"
    export_onnx(net.cpu(), testloader, output_file)

