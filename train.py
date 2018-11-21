import mnist_reader
import numpy as  np
import matplotlib.pyplot as plt
import model

#超参数
lr = 0.003
batch_size = 100
epoch_size = 100
weight_path = './weights.npz'
#weight_path = None

#网络参数
input_size=784
n_hidden_1=128
n_hidden_2=64
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
    x_test, y_test = mnist_reader.load_mnist('./data', kind='t10k')
    x_test = dataNormalize(x_test)
    y_test = transform_one_hot(y_test)
    x_train =  dataNormalize(x_train)
    y_train = transform_one_hot(y_train)

    #定义模型
    ann = model.model(input_size=input_size,n_hidden_1=n_hidden_1,n_hidden_2=n_hidden_2,weight_path=weight_path,bias=None,output_size=output_size)

    # plot 对象,画图相关
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_xlabel('epoches')
    # ax.set_ylabel('loss')
    loss = []


    for i in  range(epoch_size):
        print('----------------------------------epoch',i+1,' is running')
        # if i < 50:
        #     lr = 0.000001
        # else:
        #     lr = 0.00001
        # if (i+1) % 2 == 0:

        # 训练并返回 loss
        loss_epoch = ann.SGD(x_train,y_train,batch_size,lr)
        print('---------------------the loss of this epoch is-----------------    ' + str(loss_epoch))

        # if i % 2 == 0:
        # lr = 0.3 * lr


        # 打印 loss 变化
        loss.append(loss_epoch)
        plt.plot(loss)
        plt.show()



        # test数据的acc
        acc = ann.evaluate(x_test, y_test)
        print('acc of test data now is ' + str(acc) + " and the process is running")

        loss_test = ann.cross_entropy(ann.forward(x_test),y_test)
        print('loss of test data is ------------------------------------------    ' + str(loss_test))


        ann.saveModel()
