# AQA2024

## Prerequisites
- Linux
- Python 3.9
- PyTorch 2.3.0+cu12.0


### 安装依赖
安装依赖：
```buildoutcfg
pip install transformers datasets
pip install -U FlagEmbedding
```
如果有其他包没有安装导致报错，直接pip install即可

### 数据处理
请把train-dev数据放到AQA文件夹下，把test相关数据放到AQA-test-public文件夹下
```buildoutcfg
python clean_text/make_embedding_data.py
```

### finetune embedding
1. 微调Alibaba-NLP/gte-large-en-v1.5并推理得到每个问题top 200候选文章的推理分数
```buildoutcfg
sh first_try/gte_embedding_train.sh
```
训练完后运行，也可以不用训练直接下载checkpoint，解压后文件夹名字改为embedding_output2放在根目录下进行推理

https://drive.google.com/file/d/1iw2V2hsJPzmU_W5M0M5NVvII6eGMIes9/view?usp=sharing

```buildoutcfg
sh clean_text_test/clean_text_inference_probs.sh
```
得到AQA-test-public/result_test.jsonl，请手动重命名为result_gte.jsonl

删除embedding_output2文件夹，或者移动到别的文件夹下

2. 微调Snowflake/snowflake-arctic-embed-l并推理得到每个问题top 200候选文章的推理分数
```buildoutcfg
sh first_try/snowflake_embedding_train.sh
```
训练完后运行，也可以不用训练直接下载checkpoint，解压后文件夹名字改为embedding_output2放在根目录下进行推理

https://drive.google.com/file/d/1Hg5d0AkHxg4bAM3UAZIC8TuUoUHpeZ6Q/view?usp=sharing
```buildoutcfg
sh clean_text_test/clean_text_inference_probs.sh
```
得到AQA-test-public/result_test.jsonl，请手动重命名为result_snowflake.jsonl

删除embedding_output2文件夹，或者移动到别的文件夹下


### 模型结果融合
运行result_snowflake.jsonl和result_gte.jsonl放在根目录下，运行ensemble.py
```buildoutcfg
python ensemble.py
```

### B榜结果
|模型|B榜结果|
|:----|:----|
|Snowflake/snowflake-arctic-embed-l|0.160779090207083|
|Alibaba-NLP/gte-large-en-v1.5|0.17240293828095|
|gte+snowflake|0.184657914972311|

### 方法介绍
1. 利用FlagEmbedding finetune embedding模型
2. 对多个embedding模型进行融合
3. 使用特殊的训练数据构造技巧
