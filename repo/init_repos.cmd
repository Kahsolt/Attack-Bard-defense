@ECHO OFF

PUSHD %~dp0

git clone https://github.com/yunqing-me/AttackVLM

git clone https://github.com/xinntao/Real-ESRGAN
git clone https://github.com/cszn/SCUNet
git clone https://github.com/wyhuai/DDNM
git clone https://github.com/facebookresearch/mae
git clone https://github.com/quanlin-wu/dmae
git clone https://github.com/jiaxiaojunQAQ/Comdefend
REM git clone https://github.com/mlomnitz/DiffJPEG
git clone https://github.com/fabiobrau/local_gradients_smoothing
REM git clone https://github.com/NVlabs/DiffPure

REM this is older and in-complete
REM https://github.com/huanranchen/AdversarialAttacks
REM this is superset of NoisedDiffusionClassifiers
REM git clone https://github.com/huanranchen/NoisedDiffusionClassifiers

REM git clone https://github.com/mmSir/GainedVAE
REM git clone https://github.com/fab-jul/L3C-PyTorch
REM git clone https://github.com/Justin-Tan/high-fidelity-generative-compression

POPD

ECHO Done!
ECHO.

PAUSE
