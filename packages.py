# 刘金波
# 开发时间： 2021/10/11 19:55
# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import csv
import os

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True #每回返回的卷积算法是固定的，每次相同输入时的输出是相同的
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed) #cuda加速加载的seed
# Some Utilities
# You do not need to modify this part. 一些公共模块，不需要更改
def get_device(): #得到是否能用GPU加速的条件
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_learning_curve(loss_record, title=''): #绘图功能，绘制学习曲线
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None): #绘制DNN预测结果
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy() #列不变行变
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5) #x:target y:preds
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()
# Preprocess 预处理

class covid19dataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''

    def __init__(self,
                 path,
                 mode='train'):
        self.mode = mode #初始状况默认为train

        # Read data into numpy arrays
        with open(path) as fp:
            data_csv = list(csv.reader(fp)) #将csv中的数据转换为列表类型
            data = np.array(data_csv[1:])[:, 1:].astype(float) #，获取数值数据，将读入numpy的数据转化为浮点型np数组

        if not target_only:
            feats = list(range(93)) #40+18+18+17，由于测试集中最后一行为预测的数值，故只有93行
        else:
            # TODO: Using 40 states & 2 tested_positive features (indices = 57 & 75)
            pass

        if mode == 'test': #因为test，train，valid数据集中的数据不相同，所以在此区分
            # Testing data
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            data = data[:, feats]
            self.data = torch.FloatTensor(data) #将numpy数据类型转换为tensor类型
        else:
            # Training data (train/dev sets)
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            target = data[:, -1] #取最后一项，即标签项作为最终目标，训练集和测试集用于对比结果的’目标‘
            data = data[:, feats]

            # Splitting training data into train & dev sets #这里将训练集分为train和dev，能被10整除的进入dev，不能的进入train
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]

            # Convert data into PyTorch tensors #将数据转化为pytorch类型，这里的indices是train和dev其中之一
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features (you may remove this part to see what will happen)
        #标准化参数，将不同量纲的不同特征变为同一数量级，使得损失函数更为平滑
        #标准化的优点：提升模型的精度，提升收敛速度
        #采用均值标准化：第i维数据-第i维数据的均值/第i维数据的标准差
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1] #获取数据的列数

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index): #这是Dataset必须重写的类函数，意在按索引返回数值
        # Returns one sample at a time
        #一次返回一个样本
        if self.mode in ['train', 'dev']:
            # For training
            #用于训练
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            #用于测试，无目标
            return self.data[index]

    def __len__(self): #返回数据的行
        # Returns the size of the dataset
        #返回数据集的大小
        return len(self.data)
#DataLoader 数据加载器
def prep_dataloader(path, mode, batch_size, n_jobs=0):
    ''' Generates a dataset, then is put into a dataloader. '''
    #生成数据集，然后放入数据加载器
    dataset = covid19dataset(path, mode)  # Construct dataset，构建数据集
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False, #是否打乱数据后在读取；在数据集不是batch_size的整数倍时不考虑丢弃剩余部分
        num_workers=n_jobs, pin_memory=True)                            # Construct dataloader，构建数据加载器
    #pin_memory指的是锁页内存，将内存中的tensor转存到GPU中运行会更快一些。num_workers指的是同时进行工作的线程，默认为0，即只用主线程
    return dataloader
#Deep Neural Network DNN:NeuralNet 是一个为回归设计的 nn.Module。 DNN 由 2 个具有 ReLU 激活的全连接层组成。 该模块还包括一个用于计算损失的函数 cal_loss。
class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        # Define your neural network here，在此处定义你的神经网络
        # TODO: How to modify this model to achieve better performance?
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),#100
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Mean squared error loss（MSE）
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        #给定大小（batch_size x input_dim）的输入，计算网络的输出
        return self.net(x).squeeze(1) #对数据维度进行压缩，方便进行预测和实际值的对比

    def cal_loss(self, pred, target):
        ''' Calculate loss ''' #计算损失
        # TODO: you may implement L1/L2 regularization here
        #你可以在这里实现 L1/L2 正则化
        return self.criterion(pred, target)
#Tarining
def train(tr_set, dv_set, model, config, device):
    ''' DNN training '''

    n_epochs = config['n_epochs']  # Maximum number of epochs

    # Setup optimizer，设置优化器
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}      # for recording training loss ，记录训练损失
    early_stop_cnt = 0
    epoch = 0 #1
    while epoch < n_epochs:
        model.train()                           # set model to training mode，将模型设置为训练模式
        for x, y in tr_set:                     # iterate through the dataloader，遍历数据加载器
            optimizer.zero_grad()               # set gradient to zero，将梯度设置为零
            x, y = x.to(device), y.to(device)   # move data to device (cpu/cuda)，将数据移动到设备 (cpu/cuda)
            pred = model(x)                     # forward pass (compute output)，前向传递（计算输出）
            mse_loss = model.cal_loss(pred, y)  # compute loss，计算损失
            mse_loss.backward()                 # compute gradient (backpropagation)，计算梯度（反向传播）
            optimizer.step()                    # update model with optimizer，使用优化器更新模型，参数更新
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # After each epoch, test your model on the validation (development) set.
        #在每次循环之后，在验证（开发）集上测试您的模型。
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            # Save model if your model improved，如果您的模型有所改进，请保存模型
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path，将模型保存到指定路径
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse) #展成一维的进行记录
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            #如果您的模型在“config['early_stop']”时期停止改进，请停止训练。
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record
#Validation 验证
def dev(dv_set, model, device):
    model.eval()                                # set model to evalutation mode，将模型设置为评估模式
    total_loss = 0 #[]
    for x, y in dv_set:                         # iterate through the dataloader，遍历数据加载器
        x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)，将数据移动到设备 (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation，禁用梯度计算
            pred = model(x)                     # forward pass (compute output)，前向传递（计算输出）
            mse_loss = model.cal_loss(pred, y)  # compute loss ，计算损失
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss，累积损失
    total_loss = total_loss / len(dv_set.dataset)              # compute averaged loss，计算平均损失

    return total_loss
#Testing
def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction，收集预测
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array，连接所有预测并转换为 numpy 数组
    return preds
#Setup Hyper-parameters:设置超参数
device = get_device()                 # get the current available device ('cpu' or 'cuda')，获取当前可用的设备（'cpu' 或 'cuda'）
os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/ ，训练好的模型将保存到 ./models/
target_only = False                   # TODO: Using 40 states & 2 tested_positive features

# TODO: How to tune these hyper-parameters to improve your model's performance?
config = {
    'n_epochs': 3000,                # maximum number of epochs
    'batch_size': 270,               # mini-batch size for dataloader
    'optimizer': 'SGD',              # optimization algorithm (optimizer in torch.optim)，优化算法（torch.optim 中的优化器）
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.001,                 # learning rate of SGD，SGD：随机梯度下降
        'momentum': 0.9              # momentum for SGD
    },
    'early_stop': 200,               # early stopping epochs (the number epochs since your model's last improvement)，早期停止时期（自模型上次改进以来的时期数）
    'save_path': 'models/model.pth'  # your model will be saved here
}
#Load data and model：加载数据和模型
tr_set = prep_dataloader('covid_train.csv', 'train', batch_size=270)
dv_set = prep_dataloader('covid_train.csv', 'dev', batch_size=270)
tt_set = prep_dataloader('covid_test.csv', 'test', batch_size=270)
model = NeuralNet(tr_set.dataset.dim).to(device)  # Construct model and move to device，构建模型并移动到设备
#Start Training!
model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)
#train函数中自己带有输出部分，所以不需要冗余的输出
#学习曲线
plot_learning_curve(model_loss_record, title='deep model') #绘制图像
#真实曲线与预测曲线的比较
del model #解除引用功能，这里的model是’空指针‘，可以重新被用来引用起来的东西
model = NeuralNet(tr_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
model.load_state_dict(ckpt)
plot_pred(dv_set, model, device)  # Show prediction on the validation set，在验证集上显示预测
#Testing:The predictions of your model on testing set will be stored at pred.csv.您的模型在测试集上的预测将存储在 pred.csv 中。
def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds): #enumerate函数用于将一个可遍历的数组对象组合成一个索引序列，同时列出数据和数据下标
            writer.writerow([i, p])

preds = test(tt_set, model, device)  # predict COVID-19 cases with your model
save_pred(preds, 'pred.csv')         # save prediction file to pred.csv
