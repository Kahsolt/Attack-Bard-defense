# 引入的第三方repo，有些地方需要修改才能跑起来

    Fuck it. I mean just fuck it.

----

### 安装依赖

- `repo\init_repos.cmd`
- `pip install requirements.txt`


### Real-ESRGAN

删除文件 `repo\Real-ESRGAN\realesrgan\__init__.py` 第6行 `from .version import *`


### SCUNet

删除文件 `repo\SCUNet\models\network_scunet.py` 第6行 `from thop import profile`
