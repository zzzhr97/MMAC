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
- 删除某次训练生成的`model`和`result`：
  - 进入`control`文件夹
  - 修改`del.sh`文件中的`str`参数
  - 运行`bash del.sh`将会删除`model`和`result`文件夹中，所有包含`str`的文件

## Record
- `v0.1`
  - 上传项目文件