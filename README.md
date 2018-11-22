# ANN-pure-python-numpy


lt us build our own network without any library！


版本号2.0

Employed the RELU instead of sigmoid 

When the learning rate was 0.01. and after 100 epoch（maybe 10 mins） The loss down to 0.5, and the ACC of test data up to 82%
After we changed the LR value to 0.005 and trained 100 epoch，the ACC didn't change too much，only 83%.
![avatar](./.png)




版本号1.N

When we got 2 hidden lyaer inside the network, the acc only up to 60%, and we tried (784,512,512,10) it had the largest weights about 7MB, then is (784,512,256,10),(784,256,256,10),(784,256,128,10) which had the best performence in the tree layers network,(784,128,128,10).Sadly,they both worked not very well,though I've think the deeper and bigger network have a better ability to fit the data. And in this case ,(784,128,10)'s acc could easily up to 80%.

We took away the MSE loss，and used softmax in the last output layer with cross entropy loss.The delta of this loss × the delta of activation function just equals to y （one-hot） minus output.

版本号1.0

bug已修复，梯度下降正常

训练效果：还没咋跑

新增：权值保存


版本号0.9 

bug：梯度消失，怀疑bp算法有误，或者relu函数有问题
     应该是W初始化方法有错，正向传播时超出float范围
     未进行数据归一化，导致数据过大，溢出float范围
     

