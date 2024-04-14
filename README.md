# MMAC

## Training
- 下载数据集，将`training`和`validation`数据集分别下载到本文件夹，重命名为`Data1`和`Data2`，如`.../MMAC/Data1/2. Segmentation...`
- 超参修改
  - 在`param.py`中，修改模型名称等基础参数，可以增加或减少训练的模型
  - 在`param.py`中的`getArgs()`函数中修改总体参数，修改`default`即可
  - 在`utils.py`中的`getLrScheduler()`函数中修改学习率调度器的参数
  - 在`utils.py`中修改其他参数
- 开始训练
  - `bash train.sh`
- 会生成`model`和`result`文件夹，分别保存模型参数、训练结果

## Delete
- 删除某次训练生成的`model`，`result`和`searchlog`：
  - 进入`control`文件夹
  - 修改`del.sh`文件中的`str`，`delmodel`，`delresult`，`delsearchlog`参数
  - 运行`bash del.sh`将会删除给定文件夹中，所有包含`str`的文件
    - 比如`delmodel = 1`那么会删除`model`文件夹中所有包含`str`的文件

## Upload
- 将模型参数文件`.pth`放到`upload/upload_model/`中，三个不同的数据集分别放到对应的文件夹中
- 运行`bash test.sh`检查正确性
- 将下列文件/文件夹打包，如`1_4-12.zip`
  - `upload_model/`
  - `metadata`
  - `model.py`
- 进入比赛网站 [MMAC](https://codalab.lisn.upsaclay.fr/competitions/12476#participate-submit_results)
- 进入`future test phase`中，上传`zip`文件

## Grid Search
- `python gridsearch.py`
  - 不会输出模型，输出的日志在searchlog目录下，文件名的第一个数为使用这套超参取得最大的dice对应的（1-dice），后面是超参数

## Record
- `v0.1`
  - 上传项目文件
  - 添加`upload`文件夹，用于测试
- `v0.2`
  - 网格搜索