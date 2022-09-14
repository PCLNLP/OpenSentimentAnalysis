# MindSpore_Sentiment_Analysis

## 概述

提供情感分析相关算法的 MindSpore 实现，支持 GPU 和昇腾（HUAWEI Ascend） 双硬件平台。为所有算法提供统一接口，可通过参数设置切换不同的算法和硬件平台。依托启智社区，支持使用云脑 I 和云脑 II 资源进行线上调试和训练。PyTorch 版本的算法实现参见：https://git.openi.org.cn/PCLNLP/SentimentAnalysisNLP。

## 环境要求

- 硬件（Ascend/GPU）
  - 使用 Ascend 或 GPU 处理器来搭建硬件环境
- 框架
  - [MindSpore 1.8.1](https://www.mindspore.cn/install)

## 数据集

所有算法的数据集已上传到目录 `/dataset/sentiment_analysis_data` ，每个算法对应的数据集命名为 `算法名_data` 。以 InterGCNBERT_ABSA 为例，其对应的数据集名为 `InterGCNBERT_ABSA_data` ，该数据集的结构如下：

 ```
.
└─InterGCNBERT_ABSA_data
  ├─checkpoints									  		   
    └─ms_intergcn_bert_rest16_acc_0.9075_f1_0.7239.ckpt    # 网络正向对齐测试 ckpt      
  └─glove.42B.300d.txt       						       # 辅助数据
  ├─lap14												   # 数据集
  	├─test.raw											   # 测试集数据
  	├─test.raw.graph_af									   # 测试集图数据
  	├─test.raw.graph_inter								   # 测试集图数据
  	├─train.raw											   # 训练集数据
  	├─train.raw.graph_af								   # 训练集图数据
  	└─train.raw.graph_inter								   # 训练集图数据
  ├─rest14												   # 与 lap14 目录一致
  ├─rest15												   # 与 lap14 目录一致
  ├─rest16												   # 与 lap14 目录一致
 ```

其中以 `.ckpt` 结尾的文件为 checkpoint，该 checkpoint 可以用来测试 MindSpore 网络的正向计算是否与 PyTorch 对齐。checkpoint 的命名方式为 `框架_算法_数据集_指标_结果_...指标_结果.ckpt` 。如果要进行正向测试，只需要在 *eval* 模式加载该 checkpoint，并在对应的数据集（这里为 rest16）上进行推理即可。若推理结果在各个指标上（这里为 acc 和 f1）与 checkpoint 名字中的结果一致，则表示正向计算已对齐。 

## 算法结构（以 InterGCNBERT_ABSA 为例）

```
.
└─InterGCNBERT_ABSA
  ├─README.md
  ├─utils									  # 辅助功能代码
    ├─generate_graph_for_aspect.py            # 生成图
    ├─generate_position_con_graph.py          # 生成图
    └─init.py                                 # XavierNormal 初始化方法
  ├─__init__.py       						  # 暴露算法包的 Instructor 接口
  ├─config.yaml           					  # 配置模型参数
  ├─dataset.py          					  # 数据预处理
  ├─eval.py              					  # 网络推理流程
  ├─model.py 								  # 网络骨干编码
  └─train.py 								  # 网络训练流程
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
    --algo					# 算法包名
    --data_dir				# 该算法对应的数据集目录
    --dataset				# 具体的数据集名
    --save_ckpt_path		# ckpt 的保存路径
    --valset_ratio			# 验证集所占比例
    --bert_tokenizer		# 采用的 BERT Tokenizer
    --batch_size			# 训练和推理的 batch size
    --num_workers			# 读取数据的线程数
    --max_seq_len			# 最大序列长度
    --num_epochs			# 最大训练 epoch 数
    --warmup				# 训练中 warmup 比例
    --log_step				# 记录数据的 step 间隔数
    --patience				# EarlyStopping 的 patience
    --device				# 任务运行的目标设备，可选 Ascend、CPU 和 GPU
    --seed					# 随机数种子
    --lr					# 学习率
    --weight_decay			# 权重衰减
    --mode					# 运行模式，可选 train 和 eval
    --graph_mode			# MindSpore 图模式，若为 False，则采用 PYNATIVE_MODE
```



## 调试任务

## 训练任务

## 结果对比

## 如何贡献