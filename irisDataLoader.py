import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


class irisDataLoader(DataLoader):
    def __init__(self, path):
        try: 
            df = pd.read_csv(path, names=["sepal length","sepal width",
                                          "petal length","petal width","category"]) # 读取iris数据
        except Exception:
            print("错误路径")
        
        mapping = {
        "Iris-setosa":0,
        "Iris-versicolor":1,
        "Iris-virginica":2
        }

        df["category"] = df["category"].map(mapping)
        
        sample = df.iloc[:,:4]
        label = df.iloc[:,4:]
        
        sample = (sample - sample.mean()) / sample.std()

        self.sample = torch.from_numpy(np.array(sample, dtype='float32'))
        self.label = torch.from_numpy(np.array(label, dtype='int64')) # torch.Size([150, 1])
        # print(self.label)
        self.sampleNum = len(label)
        print(f"\033[32mirisLoader创建完毕，样本数量：{self.sampleNum}\033[0m")


    def __len__(self):
        return self.sampleNum
    

    def __getitem__(self, index):
        # self.sample = list(self.sample)
        # self.label = list(self.label)  # 好像不写也可以用索引的方式获取
        return self.sample[index], self.label[index]


if __name__ == '__main__':
    path = "iris\iris.csv"
    iris_data = irisDataLoader(path)
    sam ,lab = iris_data.__getitem__(0)
    print(sam)
    print(lab)

    


        

        
