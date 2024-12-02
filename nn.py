import torch 
import os,sys
import pandas
import numpy
from torch.utils.data import DataLoader
from irisDataLoader import irisDataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class NN(nn.Module):
    def __init__(self, input_dim, hid_dim1, hid_dim2, outp_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim,hid_dim1)
        self.layer2 = nn.Linear(hid_dim1,hid_dim2)
        self.layer3 = nn.Linear(hid_dim2,outp_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据集的划分
path = "iris\iris.csv"
iris_dataset = irisDataLoader(path)


sample_total_size = iris_dataset.__len__()

# 105 30 15
train_set_num = int(sample_total_size * 0.7)
val_set_num = int(sample_total_size * 0.2)
test_set_num = sample_total_size - train_set_num - val_set_num 

# 划分好了数量，接下来按照数量来切分
train_set, val_set, tes_set = torch.utils.data.random_split(iris_dataset, [train_set_num,
                                                                           val_set_num,test_set_num])

# 数据的加载，上面是将三个不同作用数据集的所有分堆，接下来每个堆都得按批量取出来训练和测试
# 因此需要用到dataloader，这就像一个 “吐出设备”一样，每次吐出一个batch用于训练和验证
train_loader = DataLoader(train_set, 16, True)
val_loader = DataLoader(val_set, 1, False)
tes_loader = DataLoader(tes_set, 1, False)

print(f"训练集大小：{len(train_loader) * 16}  验证集大小：{len(val_loader)}  \
测试集大小：{len(tes_loader)}")


def infer(model, dataset, device):
    model.eval()
    acc_num = 0
    acc = 0
    with torch.no_grad(): # 不希望grad有变化
        for data in dataset:
            sample, label = data
            outputs = model(sample.to(device))
            _ , pre_index = torch.max(outputs,dim=1) # 将鼠标悬浮在max上面就可以看到max函数的返回值是两个意义的数值
            acc_num += torch.eq(pre_index, label.to(device)).sum() # .item() # 这里是不是需要将label.to(device)? 果然需要！

    acc = acc_num / len(dataset)
    return acc


def main(total_epoch, lr):
    model = NN(4,5,6,3).to(device)
    loss_f = nn.CrossEntropyLoss()

    para = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(para, lr)

    save_path = os.path.join(os.getcwd(), "result_weight")

    if os.path.exists(save_path) is False:
        os.mkdir(save_path)

    
    for this_epoch in range(total_epoch):
        model.train()
        acc_num = torch.zeros(1).to(device)
        sample_number = 0

        train_bar = tqdm(train_loader, file=sys.stdout, ncols=100)

        for data in train_bar:
            sapmle, label = data
            label = label.squeeze(-1) # 这里前面的infer函数并没有squeeze为什么这里就需要squeeze？因为crossEntropy计算需要这样
            sample_number += sapmle.shape[0]
            optimizer.zero_grad()

            outputs = model(sapmle.to(device)) # [batch_size, 3]
            pre_y = torch.max(outputs,dim=1)[1] # [1] 返回类别
            acc_num += torch.eq(pre_y, label.to(device)).sum() # 那岂不是infer函数那里也得to（device）? 没错！
            loss = loss_f(outputs, label.to(device))
            loss.backward()
            optimizer.step()

            train_acc = float(acc_num / sample_number)
            train_bar.desc = "Epoch:\033[32m{}/{}\033[0m train_acc: \033[32m{:.3f}\033[0m".format(this_epoch+1,total_epoch,train_acc)

        val_acc = infer(model, val_loader, device)
        print("Epoc:{}/{} val_acc:\033[32m{:.3f}\033[0m".format(this_epoch+1,total_epoch,val_acc))
        torch.save(model.state_dict(),os.path.join(save_path,"checkPoint.pth"))

        val_acc = 0.
        # train_acc = 0. # 感觉不清零也行？ 雀食行

    print("\033[32m训练已完成!\033[0m")
    test_acc = infer(model, tes_loader ,device)
    print("\033[32m测试准确度：{:.3f}\033[0m".format(test_acc))

if __name__ == "__main__":
    main(20,0.05)