# weather_prodict
lstm_paper.py是预测气温的代码，实际代码有区别，
主要是数据的问题(本来想把特征处理上传的，但是考虑到数据量太大，就直接上传了处理后的特征)，在自己渣笔记本上运行的代码，数据没给太大，但是模型出来还不错，正确率大概70%左右,不过我用最简单的liner预测过，准确率也达到了62左右，也充分说明了算法不重要，数据特征才是最重要的==
![image](https://github.com/815670208/weather_prodict/blob/master/temperature.png)
过几天，把强天气的预测放上来，一样的算法，用的都是lstm
## 2018.6.29更新
6小时雷暴预报 从理论上来说只要有数据完全可以脱离人类
举的例子是北京，站号54511，用的差不多30年的数据，本来想把数据和模型参数一起传上来，但是实在太大，如果要数据，可以自己运行example/ec_down.py，自动下载，但是地面观测数据没有，如果要的可以留言。
搞这个的时候，参考了港科大施行健博士的论文Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting
启发很大，但是从论文上来看，效果并不能用在实践上，自己把模型稍微改了一下，效果还行，其实后面施行健博士还提出了一个TrajGRU网络，表明效果比Convolutional LSTM Network要好，这个我还没研究，等有时间还是要好好看一下的。下面这张图是测试集的混淆矩阵，预报准确率83.08%，误报率13.02%，效果还行。
![image](https://github.com/815670208/weather_prodict/blob/master/bj.png)
