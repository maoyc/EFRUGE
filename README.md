# EFRUGE
 ### Graph Encoder with Real-time Update of Edge Features based on Message Passing Mechanism

### 基于消息传递机制边特征实时更新的图编码器生成

---

**前期准备：**

1. 导入虚拟环境所需的库：requirement.txt或者environment.yaml

2. 下载TUDataset数据集（因为数据集较大，放在百度网盘自取）：

   百度网盘链接：https://pan.baidu.com/s/1_sEsqeijidanoyGlDYTWwA 
   提取码：rzat

   *PS:  TUDataset放在data文件夹内，ogbg_molhiv数据集上传网盘会报错，所以可能在运行时也会出错*

3. best_model存放训练好的模型

---

**开始运行主程序：**

1. 进入main.py主函数所在的目录

   例如：cd E:\Python_source\bookEx\File\edgeupdate

2. 按需更改main.py中文件夹目录

   ![img](https://github.com/maoyc/EFRUGE/blob/main/image/image-20220616161848491.png)

   按需调整arguments.py的参数达到不同的目的，例如：

```python
parser.add_argument('--train', dest='train', type=bool, default=True, help='')#default=True为模型训练状态，False为执行下游任务
```

​				其他参数功能，请参阅注释

3. 运行下列代码，结束后会在best_model中存放训练好的模型

```python
python main.py
```

4. 进行下游任务，在result目录中会输出结果。

```
parser.add_argument('--train', dest='train', type=bool, default=False, help='')#default=True为模型训练状态，False为执行下游任务
```

