@echo off
chcp 65001 >nul
echo ============================================================
echo 合法参数图生成器 - 快速启动
echo ============================================================
echo.

echo [步骤 1] 检查Python环境...
python --version
if errorlevel 1 (
    echo ❌ 错误: 未找到Python，请先安装Python
    pause
    exit /b 1
)
echo ✓ Python环境正常
echo.

echo [步骤 2] 检查依赖包...
python -c "import pandas, numpy, seaborn, matplotlib" 2>nul
if errorlevel 1 (
    echo ⚠️  缺少依赖包，正在安装...
    pip install pandas numpy seaborn matplotlib
    if errorlevel 1 (
        echo ❌ 依赖安装失败，请手动安装
        pause
        exit /b 1
    )
    echo ✓ 依赖安装完成
) else (
    echo ✓ 依赖包已安装
)
echo.

echo ============================================================
echo 选择运行模式:
echo ============================================================
echo 1. 快速测试（推荐首次运行）
echo 2. 正式运行（可能需要数小时）
echo 3. 退出
echo.
set /p choice="请输入选项 (1/2/3): "

if "%choice%"=="1" (
    echo.
    echo 启动快速测试...
    echo ============================================================
    python test_quick.py
    echo.
    echo 测试完成！按任意键查看测试结果...
    pause
    explorer .
) else if "%choice%"=="2" (
    echo.
    echo ⚠️  警告: 正式运行可能需要数小时，请确保:
    echo   - 有足够的磁盘空间（建议预留几GB）
    echo   - 计算机不会进入睡眠模式
    echo   - 有足够的时间等待完成
    echo.
    set /p confirm="确认继续？(y/n): "
    if /i "%confirm%"=="y" (
        echo.
        echo 启动正式运行...
        echo ============================================================
        python compute_valid_params.py
        echo.
        echo 运行完成！按任意键查看结果...
        pause
        explorer heatmaps_valid_params
    ) else (
        echo 已取消运行
    )
) else if "%choice%"=="3" (
    echo 已退出
    exit /b 0
) else (
    echo 无效选项
)

echo.
echo ============================================================
pause
