@ECHO OFF

REM configs
SET PID=0

SET RUNNER=python query_api.py -P %PID%


:clean
%RUNNER% -I dataset\NIPS17 -L 100

:adv
%RUNNER% -I dataset\ssa-cwa-200 -L 100
%RUNNER% -I outputs\adv

:defended
%RUNNER% -I outputs\blur_and_sharpen\default
%RUNNER% -I outputs\blur_and_sharpen\k=2
%RUNNER% -I outputs\comdef
%RUNNER% -I outputs\ddnm\default
%RUNNER% -I outputs\ddnm\step=2
%RUNNER% -I outputs\ddnm\step=5
%RUNNER% -I outputs\dmae\default
%RUNNER% -I outputs\jpeg\default
%RUNNER% -I outputs\mae
%RUNNER% -I outputs\realesrgan\default
%RUNNER% -I outputs\realesrgan\denoise_strength=0.8
%RUNNER% -I outputs\scunet\default
