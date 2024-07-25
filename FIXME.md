# 引入的第三方repo，有些地方需要修改才能跑起来

    Fuck it. I mean just fuck it.

----

### 安装依赖

- `repo\init_repos.cmd`
- `pip install requirements.txt`


### Real-ESRGAN

- 自行下载权重文件到 `weights` 目录
- 删除文件 `repo\Real-ESRGAN\realesrgan\__init__.py` 第6行 `from .version import *`


### SCUNet

- 下载权重文件 `python main_download_pretrained_models.py --models "SCUNet" --model_dir "model_zoo"`
- 删除文件 `repo\SCUNet\models\network_scunet.py` 第6行 `from thop import profile`


### MAE

- 参考README.md，下载对应的权重文件放在对应仓库的 `models` 的目录下 (自行创建)
- 文件 `repo\mae\util\pos_embed.py` 第 56 行 `np.float` 改为 `np.float32`


### dMAE

- 参考README.md，下载对应的权重文件放在对应仓库的 `models` 的目录下 (自行创建)


### local_grad_smooth

- 注释掉 `repo\local_gradients_smoothing\lgs\__init__.py` 第 3~5 行


### DDNM

- 参考README.md，下载权重文件 https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt 放在 `DDNM/exp/logs/imagenet/`

