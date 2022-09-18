## OpenABSA

OpenABSA 是一个以 ABSA（Aspect-Based Sentiment Analysis）算法为主的开源工具包。基于 MindSpore 实现，支持 GPU 和昇腾（HUAWEI Ascend） 双硬件平台。OpenABSA 为所有算法提供统一接口，可通过参数设置切换不同的算法和硬件平台。依托启智社区，支持使用云脑 I 和云脑 II 资源进行线上调试和训练。PyTorch 版本的算法实现参见：https://git.openi.org.cn/PCLNLP/SentimentAnalysisNLP。

## 动态

### 2022-09-09

- 目前算法包已支持 5 个 MindSpore 算法

## 关键特性

- 跨平台：同时支持 GPU 和 Ascend 硬件平台
- 多框架：同时支持 PyTorch、MindSpore 和 TensorLayer 等多个神经网络框架
- 易用性：为所有算法提供统一接口，可通过参数设置切换硬件平台和计算框架
- 易扩展：用户可自定义数据和算法，通过 Instructor API 来融入算法包

## 环境要求

- 硬件（Ascend/GPU）
  - 使用 Ascend 或 GPU 处理器来搭建硬件环境
- 框架
  - [MindSpore 1.8.1](https://www.mindspore.cn/install)
  - [PyTorch](https://pytorch.org/get-started/locally/)
  - [TensorLayer](https://tensorlayerx.readthedocs.io/en/latest/user/installation.html)

## 数据集

所有算法的数据集已上传到目录 `/dataset/sentiment_analysis_data` ，每个算法对应的数据集命名为 `算法名_data` 。以 InterGCNBERT_ABSA 为例，其对应的数据集名为 `InterGCNBERT_ABSA_data` ，该数据集的结构如下：

 ```
.
└─InterGCNBERT_ABSA_data
  ├─checkpoints   
    └─ms_intergcn_bert_rest16_acc_0.9075_f1_0.7239.ckpt     # 网络正向对齐测试 ckpt      
  └─glove.42B.300d.txt                                      # 辅助数据
  ├─lap14                                                   # 数据集
    ├─test.raw                                              # 测试集数据
    ├─test.raw.graph_af                                     # 测试集图数据
    ├─test.raw.graph_inter                                  # 测试集图数据
    ├─train.raw                                             # 训练集数据
    ├─train.raw.graph_af                                    # 训练集图数据
    └─train.raw.graph_inter                                 # 训练集图数据
  ├─rest14                                                  # 与 lap14 目录一致
  ├─rest15                                                  # 与 lap14 目录一致
  ├─rest16                                                  # 与 lap14 目录一致
 ```

其中以 `.ckpt` 结尾的文件为 checkpoint，该 checkpoint 可以用来测试 MindSpore 网络的正向计算是否与 PyTorch 对齐。checkpoint 的命名方式为 `框架_算法_数据集_指标_结果_...指标_结果.ckpt` 。如果要进行正向测试，只需要在 *eval* 模式加载该 checkpoint，并在对应的数据集（这里为 rest16）上进行推理即可。若推理结果在各个指标上（这里为 acc 和 f1）与 checkpoint 名字中的结果一致，则表示正向计算已对齐。 

## 算法结构（以 InterGCNBERT_ABSA 为例）

```
.
└─InterGCNBERT_ABSA
  ├─README.md
  ├─utils                                                   # 辅助功能代码
    ├─generate_graph_for_aspect.py                          # 生成图
    ├─generate_position_con_graph.py                        # 生成图
    └─init.py                                               # XavierNormal 初始化方法
  ├─__init__.py                                             # 暴露算法包的 Instructor 接口
  ├─config.yaml                                             # 配置模型参数
  ├─dataset.py                                              # 数据预处理
  ├─eval.py                                                 # 网络推理流程
  ├─model.py                                                # 网络骨干编码
  └─train.py                                                # 网络训练流程
```

## 统一接口

```
用法：run.py [-h] [--algo ALGO] [--data_dir DATA_DIR] [--dataset DATASET]
              [--save_ckpt_path SAVE_CKPT_PATH] [--valset_ratio VALSET_RATIO]
              [--bert_tokenizer BERT_TOKENIZER] [--batch_size BATCH_SIZE]
              [--num_workers NUM_WORKERS] [--max_seq_len MAX_SEQ_LEN]
              [--num_epochs NUM_EPOCHS] [--warmup WARMUP]
              [--log_step LOG_STEP] [--patience PATIENCE]
              [--device {Ascend,CPU,GPU}] [--seed SEED] [--lr LR]
              [--weight_decay WEIGHT_DECAY] [--mode {train,eval}]
              [--graph_mode GRAPH_MODE]
选项：
    --algo                                                  # 算法包名
    --data_dir                                              # 该算法对应的数据集目录
    --dataset                                               # 具体的数据集名
    --save_ckpt_path                                        # ckpt 的保存路径
    --valset_ratio                                          # 验证集所占比例
    --bert_tokenizer                                        # 采用的 BERT Tokenizer
    --batch_size                                            # 训练和推理的 batch size
    --num_workers                                           # 读取数据的线程数
    --max_seq_len                                           # 最大序列长度
    --num_epochs                                            # 最大训练 epoch 数
    --warmup                                                # 训练中 warmup 比例
    --log_step                                              # 记录数据的 step 间隔数
    --patience                                              # EarlyStopping 的 patience
    --device                                                # 任务运行的目标设备，可选 Ascend、CPU 和 GPU
    --seed                                                  # 随机数种子
    --lr                                                    # 学习率
    --weight_decay                                          # 权重衰减
    --mode                                                  # 运行模式，可选 train 和 eval
    --pynative_mode                                         # 使用 PYNATIVE_MODE
```



## 调试任务

首先在云脑->调试任务界面中，点击右上角的 `新建调试任务` 。进入新建任务界面后，计算资源选择 `CPU/GPU` ，镜像选择 `ms_181_cuda_111` ，数据集选择 `sentiment_analysis_data.zip` ，任务名称及资源规格按需求填写。点击新建任务，稍等片刻即可开始调试。
进入调试终端后，可以通过修改 `run.py` 改变参数，也可以直接按照统一接口处的说明选择自己想要的参数。若要在调试模式下进行训练，以InterGCN与rest16数据集为例：

```bash
python run.py --algo InterGCNBERT_ABSA --data_dir /dataset/sentiment_analysis_data/InterGCNBERT_ABSA_data --dataset rest16 --save_ckpt_path /model/checkpoints/InterGCNBERT_ABSA/rest16/best_eval.ckpt
```

## 训练任务

在云脑->训练任务界面中，点击右上角的 `新建训练任务` 。计算资源选择 `CPU/GPU` ，镜像选择 `ms_181_cuda_111` ，数据集选择 `sentiment_analysis_data.zip` ，启动文件填写 `run.py` ，任务名称、任务描述及资源规格按需求填写，运行参数按照统一接口处的说明按需求自行添加，以InterGCN与rest16数据集为例：
```
参数名          参数值
algo            InterGCNBERT_ABSA
data_dir        /dataset/sentiment_analysis_data/InterGCNBERT_ABSA_data
dataset         rest16
save_ckpt_path  /model/checkpoints/InterGCNBERT_ABSA/rest16/best_eval.ckpt
```

## 框架支持

模型 |Pytorch  |Mindspore |Tensorlayer |
|--------|------|------|--------------------|
| [AAGCN_ABSA](./AAGCN_ABSA)|:heavy_check_mark:| :heavy_check_mark: |                    |
| [InterGCNBERT_ABSA](./InterGCNBERT_ABSA)|:heavy_check_mark:| :heavy_check_mark: |                    |
| [MTST_ECE](./MTST_ECE)|:heavy_check_mark:| :heavy_check_mark: |                    |
| [Scon_ABSA](./Scon_ABSA)|:heavy_check_mark:| :heavy_check_mark: |                    |
| [SenticBERT_ABSA](./SenticBERT_ABSA)|:heavy_check_mark:| :heavy_check_mark: |                    |
| [Trans_ECE](./Trans_ECE)|:heavy_check_mark:| :heavy_check_mark: |                    |

## 结果对比

以下为各算法在论文中的提供的精度（%）与Mindspore环境下复现的训练精度（%）对比*:

### InterGCNBERT_ABSA

|数据集|论文|Mindspore|
|-----|----|:--------:|
|rest15|85.42|84.13|
|rest16|91.27|91.56|

### SenticBERT_ABSA

|数据集|论文|Mindspore|
|-----|----|:--------:|
|rest15|85.32|83.76|
|rest16|91.97|91.88|

### Scon_ABSA

|数据集|论文|Mindspore|
|-----|----|:--------:|
|rest15|85.42|84.50|
|rest16|92.53|92.37|

*:以上测试结果均为V100环境下测试所得

## 性能对比
Pytorch环境下训练平均每epoch时间：

|算法|rest14|lap14|rest15|rest16|
|--------|------|------|-------|----|
| InterGCNBERT_ABSA|31.0|20.0|10.3|15.0|
| SenticBERT_ABSA|30.4|20.0|10.2|14.9|
| Scon_ABSA|24.2|15.5|8.2|11.8|

Mindspore环境下训练平均每epoch时间**：

|算法|rest14|lap14|rest15|rest16|
|--------|------|------|-------|----|
| [InterGCNBERT_ABSA](./InterGCNBERT_ABSA)|45.2|29.0|15.0|21.6|
| [SenticBERT_ABSA](./SenticBERT_ABSA)|44.7|28.9|15.0|21.5|
| [Scon_ABSA](./Scon_ABSA)|44.3|28.5|14.7|21.4|

**:以上结果均为V100环境下测试所得