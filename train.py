import mnist_reader
import numpy as  np
import matplotlib.pyplot as plt
import model

#超参数
lr = 0.01
batch_size = 100
epoch_size = 100
#weight_path = './weights.npz'
weight_path = None

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
def visualization(loss,val_loss,acc,val_acc):
    # loss = loss
    # val_loss = val
    # acc = hist.history['acc']
    # val_acc = hist.history['val_acc']

    # make a figure
    plt.ion()
    fig = plt.figure(figsize=(8,4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss,label='train_loss')
    ax1.plot(val_loss,label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc,label='train_acc')
    ax2.plot(val_acc,label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    plt.tight_layout()
    plt.pause(1)

if __name__ == '__main__':
    x_train, y_train = mnist_reader.load_mnist('./data', kind='train')
    x_test, y_test = mnist_reader.load_mnist('./data', kind='t10k')
    x_test = dataNormalize(x_test)
    y_test = transform_one_hot(y_test)
    x_train =  dataNormalize(x_train)
    y_train = transform_one_hot(y_train)

    #定义模型
    ann = model.model(input_size=input_size,n_hidden_1=n_hidden_1,n_hidden_2=n_hidden_2,weight_path=weight_path,bias=None,output_size=output_size)
    loss = []
    test_loss = []
    acc = []
    test_acc = []

    # plot 对象,画图相关
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_xlabel('epoches')
    # ax.set_ylabel('loss')
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # plt.ion()  # 将画图模式改为交互模式






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
        loss.append(loss_epoch)
        # if i % 2 == 0:
        # lr = 0.3 * lr

        acc_train = ann.evaluate(x_train,y_train)
        acc.append(acc_train)
        # # 打印 loss 变化

        # plt.plot(loss)
        # plt.show()
        # ax.plot(acc, prediction_value, 'r-', lw=5)



        # test数据的acc
        acc_test = ann.evaluate(x_test, y_test)
        print('acc of test data now is ' + str(acc_test) + " and the process is running")
        test_acc.append(acc_test)
        # plt.plot(test_acc)
        # plt.show()



        loss_test = ann.cross_entropy(ann.forward(x_test),y_test)
        print('loss of test data is ------------------------------------------    ' + str(loss_test))
        test_loss.append(loss_test)
        # 打印 loss 变化

        visualization(loss,test_loss,acc,test_acc)
        #

        ann.saveModel()

    plt.show()
