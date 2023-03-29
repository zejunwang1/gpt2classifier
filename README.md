# gpt2classifier: 基于中文 GPT2 预训练模型的文本分类微调

gpt2classifier 展示了如何使用 transformers 的 Trainer 在中文 GPT2 预训练模型上进行文本分类任务的微调，基于 ChnSentiCorp 句子级情感分类数据集分别进行了单机单卡、单机多卡和 DeepSpeed 的训练测试。

## 1. 依赖环境

- Python 3.10.6

- torch 1.13.0

- transformers 4.27.3

- deepspeed 0.8.3

## 2. 单机单卡训练

模型训练的相关配置参数在 config/train_args.json 中。

```shell
export CUDA_VISIBLE_DEVICES=0
python train.py --train_args_file config/train_args.json --train_file data/train.tsv --eval_file data/dev.tsv --pretrained_model_name_or_path uer/gpt2-chinese-cluecorpussmall --max_seq_length 192
```

```python
{'loss': 0.7026, 'learning_rate': 1.1111111111111112e-05, 'epoch': 0.03}                                                               
{'loss': 0.6302, 'learning_rate': 2.2222222222222223e-05, 'epoch': 0.07}                                                               
{'loss': 0.4654, 'learning_rate': 3.3333333333333335e-05, 'epoch': 0.1}                                                                
{'loss': 0.2688, 'learning_rate': 4.4444444444444447e-05, 'epoch': 0.13}                                                               
{'loss': 0.3006, 'learning_rate': 4.970760233918128e-05, 'epoch': 0.17}                                                                
{'loss': 0.2728, 'learning_rate': 4.912280701754386e-05, 'epoch': 0.2}                                                                 
{'loss': 0.2221, 'learning_rate': 4.853801169590643e-05, 'epoch': 0.23}                                                                
{'loss': 0.3193, 'learning_rate': 4.7953216374269006e-05, 'epoch': 0.27}                                                               
{'loss': 0.3097, 'learning_rate': 4.736842105263158e-05, 'epoch': 0.3}                                                                 
{'loss': 0.252, 'learning_rate': 4.678362573099415e-05, 'epoch': 0.33}                                                                 
{'eval_loss': 0.2811250686645508, 'eval_accuracy': 0.8991666666666667, 'eval_precision': 0.9338235294117647, 'eval_recall': 0.8566610455311973, 'eval_f1': 0.8935795954265612, 'eval_runtime': 9.4803, 'eval_samples_per_second': 126.578, 'eval_steps_per_second': 4.008, 'epoch': 0.33}
```

在 Tesla P40 单卡上 FP32 训练耗时 13:07，验证集上的分类准确率为 0.9483。

## 3. 单机多卡训练

```shell
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 train.py --train_args_file config/train_args.json --train_file data/train.tsv --eval_file data/dev.tsv --pretrained_model_name_or_path uer/gpt2-chinese-cluecorpussmall --max_seq_length 192
```

```python
{'loss': 0.2286, 'learning_rate': 3.981264637002342e-05, 'epoch': 0.73}                                                                
{'loss': 0.1996, 'learning_rate': 3.864168618266979e-05, 'epoch': 0.8}                                                                 
{'loss': 0.1856, 'learning_rate': 3.747072599531616e-05, 'epoch': 0.87}                                                                
{'loss': 0.2194, 'learning_rate': 3.6299765807962535e-05, 'epoch': 0.93}                                                               
{'loss': 0.2114, 'learning_rate': 3.51288056206089e-05, 'epoch': 1.0}                                                                  
{'loss': 0.1148, 'learning_rate': 3.395784543325527e-05, 'epoch': 1.07}                                                                
{'loss': 0.1094, 'learning_rate': 3.2786885245901635e-05, 'epoch': 1.13}                                                               
{'loss': 0.1271, 'learning_rate': 3.1615925058548013e-05, 'epoch': 1.2}                                                                
{'loss': 0.116, 'learning_rate': 3.044496487119438e-05, 'epoch': 1.27}                                                                 
{'loss': 0.0955, 'learning_rate': 2.927400468384075e-05, 'epoch': 1.33}                                                                
{'eval_loss': 0.17813652753829956, 'eval_accuracy': 0.9391666666666667, 'eval_precision': 0.9421768707482994, 'eval_recall': 0.9342327150084317, 'eval_f1': 0.9381879762912786, 'eval_runtime': 4.8123, 'eval_samples_per_second': 249.363, 'eval_steps_per_second': 3.948, 'epoch': 1.33}
```

在 Tesla P40 两张卡上 FP32 训练耗时 06:25，验证集上的分类准确率为 0.9483。

## 4. DeepSpeed

DeepSpeed 是微软发布的一个开源深度学习训练优化库，huggingface transformers 提供了对 deepspeed 的友好集成。在 transformers 中使用 deepspeed 进行训练的关键在于 ZeRO 配置文件的编写。各配置参数的详细介绍可参考：

[DeepSpeed Configuration JSON - DeepSpeed](https://www.deepspeed.ai/docs/config-json/)

这里针对几组重要的参数进行说明：

- optimizer: 原生支持 Adam、AdamW、OneBitAdam、Lamb 和 OneBitLamb 优化器；

- scheduler: 支持 LRRangeTest、OneCycle、WarmupLR 和 WarmupDecayLR；

- ZeRO Optimizations for FP16 Training: ZeRO Optimizer 有四个不同状态：0,1,2,3，分别表示禁用、优化器状态分区、优化器+梯度状态分区、优化器+梯度+模型参数分区。实际使用的时候，我们要考虑显存占用与训练速度之间的平衡，一般来说，随着 ZeRO stage 的增大，模型训练的显存占用会减小，训练耗时增加。

### ZeRO Stage 0

模型训练的相关配置参数在 config/train_args_with_deepspeed_stage0.json 中，ZeRO 配置文件为 config/ds_stage0_config.json。

```shell
export CUDA_VISIBLE_DEVICES=0,1
deepspeed train.py --train_args_file config/train_args_with_deepspeed_stage0.json --train_file data/train.tsv --eval_file data/dev.tsv --pretrained_model_name_or_path uer/gpt2-chinese-cluecorpussmall --max_seq_length 192
```

```python
{'loss': 0.0824, 'learning_rate': 1.6510538641686182e-05, 'epoch': 2.07}                                                               
{'loss': 0.0703, 'learning_rate': 1.5339578454332553e-05, 'epoch': 2.13}                                                               
{'loss': 0.0542, 'learning_rate': 1.4168618266978923e-05, 'epoch': 2.2}                                                                
{'loss': 0.1094, 'learning_rate': 1.2997658079625294e-05, 'epoch': 2.27}                                                               
{'loss': 0.123, 'learning_rate': 1.1826697892271664e-05, 'epoch': 2.33}                                                                
{'loss': 0.0962, 'learning_rate': 1.0655737704918032e-05, 'epoch': 2.4}                                                                
{'loss': 0.1017, 'learning_rate': 9.484777517564403e-06, 'epoch': 2.47}                                                                
{'loss': 0.047, 'learning_rate': 8.313817330210773e-06, 'epoch': 2.53}                                                                 
{'loss': 0.072, 'learning_rate': 7.142857142857143e-06, 'epoch': 2.6}                                                                  
{'loss': 0.0609, 'learning_rate': 5.971896955503513e-06, 'epoch': 2.67}                                                                
{'eval_loss': 0.2272045910358429, 'eval_accuracy': 0.9433333333333334, 'eval_precision': 0.9581151832460733, 'eval_recall': 0.9258010118043845, 'eval_f1': 0.9416809605488851, 'eval_runtime': 5.264, 'eval_samples_per_second': 227.962, 'eval_steps_per_second': 3.609, 'epoch': 2.67}
```

基于`ZeRO-0`在 Tesla P40 两张卡上 FP32 训练耗时 07:04，验证集上的分类准确率为 0.9433。

### ZeRO Stage 2

模型训练的相关配置参数在 config/train_args_with_deepspeed_stage2.json 中，ZeRO 配置文件为 config/ds_stage2_config.json。

```shell
export CUDA_VISIBLE_DEVICES=0,1
deepspeed train.py --train_args_file config/train_args_with_deepspeed_stage2.json --train_file data/train.tsv --eval_file data/dev.tsv --pretrained_model_name_or_path uer/gpt2-chinese-cluecorpussmall --max_seq_length 192
```

```python
{'loss': 0.0513, 'learning_rate': 1.6510538641686182e-05, 'epoch': 2.07}                                                               
{'loss': 0.0481, 'learning_rate': 1.5339578454332553e-05, 'epoch': 2.13}                                                               
{'loss': 0.0323, 'learning_rate': 1.4168618266978923e-05, 'epoch': 2.2}                                                                
{'loss': 0.0391, 'learning_rate': 1.2997658079625294e-05, 'epoch': 2.27}                                                               
{'loss': 0.0547, 'learning_rate': 1.1826697892271664e-05, 'epoch': 2.33}                                                               
{'loss': 0.043, 'learning_rate': 1.0655737704918032e-05, 'epoch': 2.4}                                                                 
{'loss': 0.0942, 'learning_rate': 9.484777517564403e-06, 'epoch': 2.47}                                                                
{'loss': 0.0386, 'learning_rate': 8.313817330210773e-06, 'epoch': 2.53}                                                                
{'loss': 0.0334, 'learning_rate': 7.142857142857143e-06, 'epoch': 2.6}                                                                 
{'loss': 0.0436, 'learning_rate': 5.971896955503513e-06, 'epoch': 2.67}                                                                
{'eval_loss': 0.19890624284744263, 'eval_accuracy': 0.9508333333333333, 'eval_precision': 0.9635416666666666, 'eval_recall': 0.9359190556492412, 'eval_f1': 0.9495295124037638, 'eval_runtime': 5.493, 'eval_samples_per_second': 218.459, 'eval_steps_per_second': 3.459, 'epoch': 2.67}
```

基于`ZeRO-2`在 Tesla P40 两张卡上 FP32 训练耗时 09:09，验证集上的分类准确率为 0.9508。

### ZeRO Stage 3

模型训练的相关配置参数在 config/train_args_with_deepspeed_stage3.json 中，ZeRO 配置文件为 config/ds_stage3_config.json。

```shell
export CUDA_VISIBLE_DEVICES=0,1
deepspeed train.py --train_args_file config/train_args_with_deepspeed_stage3.json --train_file data/train.tsv --eval_file data/dev.tsv --pretrained_model_name_or_path uer/gpt2-chinese-cluecorpussmall --max_seq_length 192
```

```python
{'loss': 0.0557, 'learning_rate': 1.6510538641686182e-05, 'epoch': 2.07}                                                               
{'loss': 0.061, 'learning_rate': 1.5339578454332553e-05, 'epoch': 2.13}                                                                
{'loss': 0.0294, 'learning_rate': 1.4168618266978923e-05, 'epoch': 2.2}                                                                
{'loss': 0.0603, 'learning_rate': 1.2997658079625294e-05, 'epoch': 2.27}                                                               
{'loss': 0.0655, 'learning_rate': 1.1826697892271664e-05, 'epoch': 2.33}                                                               
{'loss': 0.0392, 'learning_rate': 1.0655737704918032e-05, 'epoch': 2.4}                                                                
{'loss': 0.0933, 'learning_rate': 9.484777517564403e-06, 'epoch': 2.47}                                                                
{'loss': 0.0354, 'learning_rate': 8.313817330210773e-06, 'epoch': 2.53}                                                                
{'loss': 0.0425, 'learning_rate': 7.142857142857143e-06, 'epoch': 2.6}                                                                 
{'loss': 0.0409, 'learning_rate': 5.971896955503513e-06, 'epoch': 2.67}                                                                
{'eval_loss': 0.18973714113235474, 'eval_accuracy': 0.9475, 'eval_precision': 0.9616724738675958, 'eval_recall': 0.9308600337268128, 'eval_f1': 0.9460154241645243, 'eval_runtime': 6.0128, 'eval_samples_per_second': 199.574, 'eval_steps_per_second': 3.16, 'epoch': 2.67}
```

基于`ZeRO-3`在 Tesla P40 两张卡上 FP32 训练耗时 10:32，验证集上的分类准确率为 0.9475。

## 5. 显存比较

设定`per_device_train_batch_size=32`,`per_device_eval_batch_size=32`，不同训练方式的单卡显存占用与训练速度比较如下表所示：

| 训练方式    | 单卡     | DDP    | ZeRO-0 | ZeRO-2 | ZeRO-3 |
|:-------:|:------:|:------:|:------:|:------:|:------:|
| 显存占用/MB | 10755  | 11245  | 6597   | 6641   | 5659   |
| 训练耗时/s  | 787    | 385    | 424    | 549    | 632    |
| 验证集准确率  | 0.9483 | 0.9483 | 0.9433 | 0.9508 | 0.9475 |

## 6. Contact

邮箱：[wangzejunscut@126.com](mailto:wangzejunscut@126.com)

微信：autonlp


