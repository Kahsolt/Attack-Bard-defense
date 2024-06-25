# 实验手册

⚠ 我感觉我总在做一些弱智保姆软件。    

----

### 操作流程概述

#### 产生各个数据集

- 两个现成的数据集不用生成
  - 原图: [dataset\NIPS17](dataset\NIPS17)
  - 对抗样本 (预制): [dataset\ssa-cwa-200](dataset\ssa-cwa-200)
- 产生对抗样本
  - run `attack_img_encoder_misdescription.py --dfn none -O <path/to/output/folder>`
- 产生对抗样本 (带防御)
  - run `attack_img_encoder_misdescription.py --dfn blur_and_sharpen|realesrgan|scunet|ddnm|mae|dmae|comdef|jpeg|lgs -O <path/to/output/folder>`

⚠ 按约定，将数据集生成到 `outputs/<dfn>/<version>` 目录下*
  - 防御模型不带参数的情况，无 version 子目录: `outputs/adv`, `outputs/mae`
  - 防御模型带参数的情况，有 version 子目录: `outputs/blur_and_sharpen/default`, `outputs/blur_and_sharpen/k=2`


#### 评价指标结果: 图-图

> 图像相似性度量评价

- run `python stats_img_dist.py -O <path/to/img/folder> -S <output.json>`
  - e.g.: `python stats_img_dist.py -O outputs/adv`


#### 评价指标结果: 文-文

> 图像识别后的文本相似性度量评价

⚪ 首先跑图生文模型（百度fuyu-8b），或者主动收集图像描述文本 (**暂未实现**)

- 配置 AUTH 信息
  - 复制文件 `API_KEY.yaml.example` 并重命名为 `API_KEY.yaml.example`
  - 按照文件内容的提示填入 `API_KEY` 和 `SECRET_KEY`，删除 `API_EP` 这一行
- run `python query_api.py -I <path/to/img/folder> -P <pid> -L <limit>`
  - e.g.: `python query_api.py -I outputs/adv -P 0 -L 100`

⚪ 然后跑文本相似度模型

- run `python stats_txt_sim.py -M <model> -AX <path/to/img/folder>`
  - e.g.: `python stats_txt_sim.py -M all-MiniLM-L6-v2 -AX outputs/adv`


### 操作流程最小案例

⚠ 约定: 生成的图放在 outputs 目录下，图-文查询结果放在 log/record.db 里，生成的评价指标统计结果放在 log 目录下

```bash
# run attack (w/o. defense) on first 4 samples
python attack_img_encoder_misdescription.py --dfn none --limit 4 --output outputs/test
# run img-sim eval
python stats_img_dist.py -O outputs/test
# run online query img2txt
python query_api.py -I outputs/test
# run txt-sim eval
python stats_txt_sim.py -M all-MiniLM-L6-v2 -AX outputs/test
```

----
by Armit
周二 2024/06/25 
