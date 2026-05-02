@echo off
chcp 65001 >nul 2>&1
title 农险期货智能定价系统 v3.0 - 一键启动
color 0A
echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║     🌾 农险期货智能定价系统 v3.0 — 特等奖冲刺版              ║
echo  ║     QL大脑 V135 ASI 驱动 | 三较三审通过                      ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.
cd /d "%~dp0"

echo  [1/4] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo  ❌ 未检测到Python，请先安装Python 3.10+
    pause & exit /b 1
)
echo  ✅ Python环境正常

echo.
echo  [2/4] 安装依赖(首次运行)...
pip install -q streamlit plotly pandas numpy scipy scikit-learn xgboost lightgbm fastapi uvicorn python-docx openpyxl 2>nul

echo.
echo  [3/4] 启动双服务...
echo    🖥️  Streamlit UI: http://localhost:8501
echo    📡  FastAPI API:  http://localhost:8000/docs
echo.

start "Streamlit-UI" cmd /c "streamlit run src/app.py --server.port 8501 --server.headled true"
timeout /t 4 /nobreak >nul
start "FastAPI-API" cmd /c "set PYTHONPATH=.\src && python -m uvicorn api:app --host 0.0.0.0 --port 8000"

echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║  ✅ 系统启动完成!                                           ║
echo  ║                                                              ║
echo  ║  📱 可视化界面: http://localhost:8501                        ║
echo  ║  📡 API文档(Swagger): http://localhost:8000/docs             ║
echo  ║  📡 API文档(ReDoc):   http://localhost:8000/redoc            ║
echo  ║                                                              ║
echo  ║  按 Ctrl+C 停止服务                                         ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.

:loop
timeout /t 3600 /nobreak >nul
goto loop
