DEL /Q/S .\src
DEL /Q/S .\binding
DEL /Q/S .\scripts
DEL /Q binding.gyp

Xcopy /E ..\tfjs-node\src .\src
Xcopy /E ..\tfjs-node\binding .\binding
Xcopy /E ..\tfjs-node\scripts .\scripts
Xcopy ..\tfjs-node\binding.gyp .
