# EFRUGE
 Graph Encoder with Real-time Update of Edge Features based on Message Passing Mechanism

基于消息传递机制边特征实时更新的图编码器生成

1.

导入虚拟环境所需的库：requirement.txt或者environment.yaml

2.

进入main.py主函数所在的目录

例如：cd E:\Python_source\bookEx\File\edgeupdate

3.

按需更改main.py中文件夹目录

![img](https://github.com/maoyc/EFRUGE/blob/main/image/image-20220616161848491.png)

按需调整arguments.py的参数达到不同的目的，例如：

```python
parser.add_argument('--train', dest='train', type=bool, default=True, help='')
```

此时为模型训练状态

4.

```python
python main.py
```

5.

在best_model会存储最好的模型，进行下游任务，在result目录中会存储结果。
