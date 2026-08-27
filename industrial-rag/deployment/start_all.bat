@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Day21 Stage3 - Industrial RAG unified startup (Windows compatibility)
rem Location: industrial-rag\deployment\start_all.bat
rem Windows does not have to run the full GPU stack today. If matching parent
rem start_*.bat launchers exist they are used; otherwise those services are
rem reported as SKIP and the Industrial RAG API can still be started/degraded.

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"
for %%I in ("%PROJECT_ROOT%\..") do set "PLATFORM_ROOT=%%~fI"

if not defined RAG_CONFIG_FILE set "RAG_CONFIG_FILE=%PROJECT_ROOT%\config\local.env"
if not defined RAG_CONDA_ENV set "RAG_CONDA_ENV=rag_system"
if not defined LLM_CONDA_ENV set "LLM_CONDA_ENV=vllm_qwen3"
if not defined EMBEDDING_CONDA_ENV set "EMBEDDING_CONDA_ENV=embedding_service"
if not defined RERANKER_CONDA_ENV set "RERANKER_CONDA_ENV=reranker_service"
if not defined GATEWAY_CONDA_ENV set "GATEWAY_CONDA_ENV=llm_gateway"

if /I "%~1"=="--help" goto :help
if /I "%~1"=="-h" goto :help
if /I "%~1"=="--check-port" goto :check_port_cli

if not exist "%RAG_CONFIG_FILE%" (
    echo [ERROR] Config file not found: %RAG_CONFIG_FILE%
    exit /b 1
)

call :load_env "%RAG_CONFIG_FILE%"
if errorlevel 1 exit /b 1

call :require LLM_BASE_URL
if errorlevel 1 exit /b 1
call :require EMBEDDING_BASE_URL
if errorlevel 1 exit /b 1
call :require RERANKER_BASE_URL
if errorlevel 1 exit /b 1
call :require LLM_PORT
if errorlevel 1 exit /b 1
call :require EMBEDDING_PORT
if errorlevel 1 exit /b 1
call :require RERANKER_PORT
if errorlevel 1 exit /b 1
call :require GATEWAY_PORT
if errorlevel 1 exit /b 1
call :require RAG_PORT
if errorlevel 1 exit /b 1
call :require MODEL_DIR
if errorlevel 1 exit /b 1
call :require LOG_DIR
if errorlevel 1 exit /b 1

call :validate_port LLM_PORT !LLM_PORT!
if errorlevel 1 exit /b 1
call :validate_port EMBEDDING_PORT !EMBEDDING_PORT!
if errorlevel 1 exit /b 1
call :validate_port RERANKER_PORT !RERANKER_PORT!
if errorlevel 1 exit /b 1
call :validate_port GATEWAY_PORT !GATEWAY_PORT!
if errorlevel 1 exit /b 1
call :validate_port RAG_PORT !RAG_PORT!
if errorlevel 1 exit /b 1

call :check_duplicate_ports
if errorlevel 1 exit /b 1

if /I "%~1"=="--validate" goto :validate_only
if not "%~1"=="" (
    echo [ERROR] Unknown argument: %~1
    goto :help_error
)

call :check_port "LLM" !LLM_PORT!
if errorlevel 1 exit /b 1
call :check_port "Embedding" !EMBEDDING_PORT!
if errorlevel 1 exit /b 1
call :check_port "Reranker" !RERANKER_PORT!
if errorlevel 1 exit /b 1
call :check_port "Gateway" !GATEWAY_PORT!
if errorlevel 1 exit /b 1
call :check_port "Industrial RAG" !RAG_PORT!
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo Day21 Stage3 ^| Industrial RAG Unified Startup ^(Windows^)
echo ============================================================
echo Config: %RAG_CONFIG_FILE%
echo Project: %PROJECT_ROOT%
echo Platform: %PLATFORM_ROOT%
echo.

rem Optional future Windows launchers. Absence is not an error today.
call :start_optional "LLM" "%PLATFORM_ROOT%\deployment\start_llm.bat"
if not errorlevel 2 call :wait_http "LLM" "!LLM_BASE_URL!/v1/models" 60

call :start_optional "Embedding" "%PLATFORM_ROOT%\deployment\start_embedding.bat"
if not errorlevel 2 call :wait_http "Embedding" "!EMBEDDING_BASE_URL!/health" 60

call :start_optional "Reranker" "%PLATFORM_ROOT%\deployment\start_reranker.bat"
if not errorlevel 2 call :wait_http "Reranker" "!RERANKER_BASE_URL!/health" 60

call :start_optional "Gateway" "%PLATFORM_ROOT%\deployment\start_gateway.bat"
if not errorlevel 2 call :wait_http "Gateway" "http://127.0.0.1:!GATEWAY_PORT!/health" 30

where conda >nul 2>&1
if errorlevel 1 (
    echo [WARN] conda not found. Industrial RAG was not started.
) else (
    echo [INFO] Starting Industrial RAG in conda env: %RAG_CONDA_ENV%
    start "Industrial RAG" /D "%PROJECT_ROOT%" cmd /c ^
        "set RAG_CONFIG_FILE=%RAG_CONFIG_FILE%&& set RAG_ENV=local&& conda run --no-capture-output -n %RAG_CONDA_ENV% python -m uvicorn app.main:app --host 0.0.0.0 --port !RAG_PORT!"
    call :wait_http "Industrial RAG" "http://127.0.0.1:!RAG_PORT!/health" 30
)

echo.
echo ============================================================
echo Final status
echo ============================================================
call :probe_http "LLM" "!LLM_BASE_URL!/v1/models"
call :probe_http "Embedding" "!EMBEDDING_BASE_URL!/health"
call :probe_http "Reranker" "!RERANKER_BASE_URL!/health"
call :probe_http "Gateway" "http://127.0.0.1:!GATEWAY_PORT!/health"
call :probe_http "Industrial RAG" "http://127.0.0.1:!RAG_PORT!/health"
echo ============================================================
exit /b 0

:validate_only
where powershell >nul 2>&1 || (
    echo [ERROR] PowerShell not found.
    exit /b 1
)
if exist "%PROJECT_ROOT%\app\main.py" (
    echo [OK] Industrial RAG entry: %PROJECT_ROOT%\app\main.py
) else (
    echo [ERROR] Industrial RAG entry not found: %PROJECT_ROOT%\app\main.py
    exit /b 1
)
if exist "%MODEL_DIR%" (
    echo [OK] MODEL_DIR: %MODEL_DIR%
) else if exist "%PROJECT_ROOT%\%MODEL_DIR%" (
    echo [OK] MODEL_DIR: %PROJECT_ROOT%\%MODEL_DIR%
) else (
    echo [WARN] MODEL_DIR does not exist on this Windows machine: %MODEL_DIR%
)
echo [OK] Windows compatibility validation passed. No service was started.
exit /b 0

:load_env
for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%~1") do (
    if not "%%A"=="" set "%%A=%%B"
)
exit /b 0

:require
if not defined %~1 (
    echo [ERROR] Missing required config: %~1
    exit /b 1
)
exit /b 0

:validate_port
set "_PORT_NAME=%~1"
set "_PORT_VALUE=%~2"
for /f "delims=0123456789" %%A in ("%_PORT_VALUE%") do (
    echo [ERROR] %_PORT_NAME% must be an integer: %_PORT_VALUE%
    exit /b 1
)
if %_PORT_VALUE% LSS 1 (
    echo [ERROR] %_PORT_NAME% must be in 1..65535: %_PORT_VALUE%
    exit /b 1
)
if %_PORT_VALUE% GTR 65535 (
    echo [ERROR] %_PORT_NAME% must be in 1..65535: %_PORT_VALUE%
    exit /b 1
)
exit /b 0

:check_duplicate_ports
set "_P1=!LLM_PORT!"
set "_P2=!EMBEDDING_PORT!"
set "_P3=!RERANKER_PORT!"
set "_P4=!GATEWAY_PORT!"
set "_P5=!RAG_PORT!"
for %%A in (!_P1! !_P2! !_P3! !_P4! !_P5!) do (
    set /a _COUNT=0
    for %%B in (!_P1! !_P2! !_P3! !_P4! !_P5!) do if "%%A"=="%%B" set /a _COUNT+=1
    if !_COUNT! GTR 1 (
        echo [ERROR] Duplicate configured port: %%A
        exit /b 1
    )
)
exit /b 0

:check_port
set "_SERVICE=%~1"
set "_PORT=%~2"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$x = Get-NetTCPConnection -State Listen -LocalPort %_PORT% -ErrorAction SilentlyContinue; if ($x) { exit 1 } else { exit 0 }" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Port %_PORT% is already in use.
    echo Service: %_SERVICE%
    exit /b 1
)
echo [OK] %_SERVICE% port %_PORT% is available.
exit /b 0

:check_port_cli
if "%~2"=="" goto :help_error
if "%~3"=="" goto :help_error
call :check_port "%~2" "%~3"
exit /b %errorlevel%

:start_optional
set "_SERVICE=%~1"
set "_SCRIPT=%~2"
if exist "%_SCRIPT%" (
    echo [INFO] Starting %_SERVICE% using %_SCRIPT%
    start "%_SERVICE%" cmd /c ""%_SCRIPT%""
) else (
    echo [SKIP] %_SERVICE% Windows launcher not found: %_SCRIPT%
    exit /b 2
)
exit /b 0

:wait_http
set "_SERVICE=%~1"
set "_URL=%~2"
set "_SECONDS=%~3"
for /L %%S in (1,1,%_SECONDS%) do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "try { $r = Invoke-WebRequest -UseBasicParsing -Uri '%_URL%' -TimeoutSec 2; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 300) { exit 0 } } catch {}; exit 1" >nul 2>&1
    if not errorlevel 1 (
        echo [READY] %_SERVICE% %_URL%
        exit /b 0
    )
    >nul timeout /t 1 /nobreak
)
echo [WARN] %_SERVICE% is not ready: %_URL%
exit /b 0

:probe_http
set "_SERVICE=%~1"
set "_URL=%~2"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { $r = Invoke-WebRequest -UseBasicParsing -Uri '%_URL%' -TimeoutSec 3; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 300) { exit 0 } } catch {}; exit 1" >nul 2>&1
if errorlevel 1 (
    echo %_SERVICE%  DOWN   %_URL%
) else (
    echo %_SERVICE%  READY  %_URL%
)
exit /b 0

:help
echo Usage:
echo   deployment\start_all.bat
echo   deployment\start_all.bat --validate
echo   deployment\start_all.bat --check-port SERVICE PORT
exit /b 0

:help_error
call :help
exit /b 2
