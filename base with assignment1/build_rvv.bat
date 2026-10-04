@echo off
REM ============================================================
REM  rdFFT RVV 编译 & 模拟运行脚本 (Windows → WSL2)
REM
REM  VS 集成方式:
REM   项目属性 → 生成事件 → 生成后事件 → 命令行:
REM   "$(ProjectDir)build_rvv.bat"
REM ============================================================

set PROJECT_DIR=/mnt/c/Users/wenkk/Desktop/dachuang

echo [1/2] Cross-compiling RVV code...
wsl -d Ubuntu-24.04 -- bash -c "cd %PROJECT_DIR% && make clean && make 2>&1"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Compilation failed!
    exit /b 1
)

echo.
echo [2/2] Running on RISC-V emulator (QEMU user-mode)...
echo ----------------------------------------
wsl -d Ubuntu-24.04 -- bash -c "cd %PROJECT_DIR% && make run 2>&1"
echo ----------------------------------------

exit /b %ERRORLEVEL%
