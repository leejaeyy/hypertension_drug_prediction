@echo off
chcp 65001 > nul
echo =============================================
echo  고혈압 약물 추천 시스템 실행 (Streamlit)
echo =============================================
echo.
echo [1] Ollama 서버가 실행 중인지 확인하세요
echo     실행 안됐으면 새 창에서: ollama serve
echo.
echo [2] 앱을 시작합니다...
echo.
cd /d "%~dp0"
streamlit run app.py --server.port 8501
pause
