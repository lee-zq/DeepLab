import torchvision
from torchvision import transforms
import torch
from tqdm import tqdm

classes_cifar10 = ('plane', 'car', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck')

class CIFARDataset():
    def __init__(self, dataset_path, batchsize, num_workers=0):
        self.batch_size = batchsize
        self.dataset_path = dataset_path
        # self.mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        # self.std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        self.num_workers= num_workers
        self.transform_train = transforms.Compose([# 数据增强
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomGrayscale(0.15),
                                    transforms.RandomAffine((-30,30)),
                                    transforms.RandomRotation(20),  
                                    transforms.ToTensor(),
                                    transforms.Normalize(self.mean, self.std),
        ])
        self.transform_test = transforms.Compose(
                                    [transforms.ToTensor(),
                                    transforms.Normalize(self.mean, self.std)])
    def get_cifar10_dataloader(self):
        cifar10_training = torchvision.datasets.CIFAR10(root=self.dataset_path, train=True, download=True, transform=self.transform_train)
        trainloader = torch.utils.data.DataLoader(cifar10_training, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            
        cifar10_testing = torchvision.datasets.CIFAR10(root=self.dataset_path, train=False, download=False, transform=self.transform_test)
        testloader = torch.utils.data.DataLoader(cifar10_testing, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        return trainloader,testloader 
    
    def get_cifar100_dataloader(self):
        cifar100_training = torchvision.datasets.CIFAR100(root=self.dataset_path, train=True, download=True, transform=self.transform_train)
        trainloader = torch.utils.data.DataLoader(cifar100_training, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            
        cifar100_testing = torchvision.datasets.CIFAR100(root=self.dataset_path, train=False, download=False, transform=self.transform_test)
        testloader = torch.utils.data.DataLoader(cifar100_testing, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        return trainloader,testloader 

         

# test
if "__main__"==__name__:
    CIFAR_PATH = "D:/Files/projects/vs/DeepInfer/data"
    dataset = CIFARDataset(CIFAR_PATH, 4)
    train_dataloader, _ = dataset.cifar100_dataloader()

    for idx, (data,label) in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
        print(data.shape, len(label))