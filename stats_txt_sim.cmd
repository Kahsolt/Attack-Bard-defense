@ECHO OFF

REM configs
SET MODEL=all-MiniLM-L6-v2

SET RUNNER=python stats_txt_sim.py -M %MODEL%

:adv
%RUNNER% -AX dataset\ssa-cwa-200
%RUNNER% -AX outputs\adv

:defended
%RUNNER% -AX outputs\blur_and_sharpen\default
%RUNNER% -AX outputs\blur_and_sharpen\k=2
%RUNNER% -AX outputs\comdef
%RUNNER% -AX outputs\ddnm\default
%RUNNER% -AX outputs\ddnm\step=2
%RUNNER% -AX outputs\ddnm\step=5
%RUNNER% -AX outputs\dmae\default
%RUNNER% -AX outputs\jpeg\default
%RUNNER% -AX outputs\mae
%RUNNER% -AX outputs\realesrgan\default
%RUNNER% -AX outputs\realesrgan\denoise_strength=0.8
%RUNNER% -AX outputs\scunet\default
