import mnist_reader
import numpy as  np
import model

#超参数
lr = 0.0001
batch_size = 100
epoch_size = 100
weight_path = './weights.npz'
# weight_path = None

#网络参数
input_size=784
n_hidden=512
output_size=10


#装载数据
def transform_one_hot(labels):
  n_labels = np.max(labels) + 1
  one_hot = np.eye(n_labels)[labels]
  return one_hot
def dataNormalize(data):
    max=data.max()
    min=data.min()
    data= (data-min)/(min+max)
    return  data


if __name__ == '__main__':
    x_train, y_train = mnist_reader.load_mnist('./data', kind='train')
    #x_test, y_test = mnist_reader.load_mnist('./data', kind='t10k')
    x_train =  dataNormalize(x_train)
    y_train = transform_one_hot(y_train)

    #定义模型
    ann = model.model(input_size=input_size,n_hidden=n_hidden,output_size=output_size,weight_path=weight_path,bias=None)

    for i in  range(epoch_size):
        print('----------------------------------epoch',i+1,' is running')
        # if i < 50:
        #     lr = 0.000001
        # else:
        #     lr = 0.00001
        ann.SGD(x_train,y_train,batch_size,lr)
        if i % 2 == 0:
            ann.saveModel()
