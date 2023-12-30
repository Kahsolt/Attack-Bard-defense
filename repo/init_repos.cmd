@ECHO OFF

PUSHD %~dp0

git clone https://github.com/xinntao/Real-ESRGAN
git clone https://github.com/cszn/SCUNet
git clone https://github.com/wyhuai/DDNM

POPD

ECHO Done!
ECHO.

PAUSE
