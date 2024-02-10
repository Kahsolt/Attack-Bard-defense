@ECHO OFF

PUSHD %~dp0

git clone https://github.com/xinntao/Real-ESRGAN
git clone https://github.com/cszn/SCUNet
git clone https://github.com/wyhuai/DDNM
git clone https://github.com/facebookresearch/mae
git clone https://github.com/quanlin-wu/dmae

git clone https://github.com/jiaxiaojunQAQ/Comdefend

POPD

ECHO Done!
ECHO.

PAUSE
