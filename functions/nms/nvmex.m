function nvmex(cuFileName, outDir)
%NVMEX Compiles and links a CUDA file for MATLAB usage
% NVMEX(FILENAME) will create a MEX-File (also with the name FILENAME) by
% invoking the CUDA compiler, nvcc, and then linking with the MEX
% function in MATLAB.

if ispc % Windows
 Host_Compiler_Location = '-ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64"';
 CUDA_INC_Location = ['"' getenv('CUDA_PATH')  '\include"'];
    CUDA_SAMPLES_Location =['"' getenv('NVCUDASAMPLES6_5_ROOT')  '\common\inc"'];
    PIC_Option = '';
    if ( strcmp(computer('arch'),'win32') ==1)
        machine_str = ' --machine 32 ';
        CUDA_LIB_Location = ['"' getenv('CUDA_PATH')  '\lib\Win32"'];
    elseif  ( strcmp(computer('arch'),'win64') ==1)
        machine_str = ' --machine 64 ';
        CUDA_LIB_Location = ['"' getenv('CUDA_PATH')  '\lib\x64"'];
    end
    NVCC = 'nvcc';
else % Mac and Linux (assuming gcc is on the path)
    CUDA_INC_Location = '/usr/local/cuda/include';
    CUDA_SAMPLES_Location = '/usr/local/cuda/samples/common/inc';
    Host_Compiler_Location = ' ';
    PIC_Option = ' --compiler-options -fPIC ';
    machine_str = [];
    CUDA_LIB_Location = '/usr/local/cuda/lib64';
    NVCC = '/usr/local/cuda/bin/nvcc';
end
% !!! End of things to modify !!!
[~, filename] = fileparts(cuFileName);
nvccCommandLine = [ ...
NVCC ' --compile ' Host_Compiler_Location ' ' ...
'-o '  filename '.o ' ...
machine_str PIC_Option ...
' -I' '"' matlabroot '/extern/include "' ...
' -I' CUDA_INC_Location ' -I' CUDA_SAMPLES_Location ...
' "' cuFileName '" ' 
 ];
mexCommandLine = ['mex ' '-outdir ' outDir ' ' filename '.o'  ' -L' CUDA_LIB_Location  ' -lcudart'];
disp(nvccCommandLine);
warning off;
status = system(nvccCommandLine);
warning on;
if status < 0
 error 'Error invoking nvcc';
end
disp(mexCommandLine);
eval(mexCommandLine);
end
