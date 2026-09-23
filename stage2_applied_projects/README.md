# 第二阶段：应用型深度学习流程

第二阶段在第一阶段训练闭环的基础上，增加新的任务类型、数据组织方式和模型改造能力。每个项目仍保留完整流程：

```text
imports -> config -> public dataset -> Dataset/DataLoader -> model
-> loss/optimizer -> training -> evaluation -> checkpoint -> inference
```

## 项目列表

| 编号 | 项目 | 公开数据 | 重点 |
|---|---|---|---|
| 08 | CNN 图像回归 | MNIST | 连续输出、MSE、MAE、RMSE |
| 09 | U-Net 图像分割 | Oxford-IIIT Pet | image/mask、skip connection、BCE、Dice |
| 10 | Embedding + LSTM 文本分类 | UCI SMS Spam Collection | tokenize、vocabulary、padding、二分类 |
| 11 | ResNet18 迁移学习 | CIFAR-10 | 冻结、替换分类头、分组学习率、fine-tuning |
| 12 | 多步时序预测 | ETTh1 | 多变量窗口、未来24步、逆标准化、MAE/RMSE |

> 08 使用 MNIST 数字值作为连续回归目标，目的是专门比较分类头与回归头。它是教学练习，不代表识别数字时回归优于分类。

## 下载数据

在仓库根目录运行：

```bash
python stage2_applied_projects/download_data/download_08_mnist.py
python stage2_applied_projects/download_data/download_09_oxford_pet.py
python stage2_applied_projects/download_data/download_10_sms_spam.py
python stage2_applied_projects/download_data/download_11_cifar10.py
python stage2_applied_projects/download_data/download_12_etth1.py
```

## 运行项目

```bash
python stage2_applied_projects/08_cnn_image_regression.py
python stage2_applied_projects/09_unet_image_segmentation.py
python stage2_applied_projects/10_lstm_text_classification.py
python stage2_applied_projects/11_transfer_learning_cifar10.py
python stage2_applied_projects/12_multistep_time_series_forecast.py
```

可以临时减少训练轮数做快速检查：

```powershell
$env:EPOCHS=1
python stage2_applied_projects/09_unet_image_segmentation.py
```

迁移学习分别使用 `HEAD_EPOCHS` 和 `FINETUNE_EPOCHS`。

## 每个项目的练习方式

1. 第一天运行标准代码，标出所有关键 tensor shape。
2. 第二天只保留流程标题，从空白文件重写。
3. 第三天更换一个条件，例如输出维度、loss、backbone 或预测长度。
4. 第四天记录实验配置、验证指标、错误原因和解决方法。

完成标准：能够独立修改任务，并解释为什么数据划分、loss、metric 和预测头必须随任务变化。
