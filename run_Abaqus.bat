@echo off
setlocal

if "%~1"=="" (
    echo.
    echo Usage: run_Abaqus.bat CPU_COUNT GPU_COUNT MODEL_NAME
    echo        run_Abaqus.bat MODEL_NAME
    echo.
    exit /b 1
)

if not "%~3"=="" (
    set "CPU=%~1"
    set "GPU=%~2"
    set "NAME=%~3"
) else (
    set "CPU=0"
    set "GPU=0"
    set "NAME=%~1"
)

if "%NAME%"=="" (
    echo ERROR: MODEL_NAME is required.
    echo Usage: run_Abaqus.bat CPU_COUNT GPU_COUNT MODEL_NAME
    exit /b 1
)

set "ABAQUS_OPTIONS="
set "PRECISION_OPTION=double=both"
if not "%CPU%"=="0" set "ABAQUS_OPTIONS=%ABAQUS_OPTIONS% cpus=%CPU%"
if not "%GPU%"=="0" (
    set "ABAQUS_OPTIONS=%ABAQUS_OPTIONS% gpus=%GPU% parallel=domain"
    set "PRECISION_OPTION="
)

set "ROOT=%~dp0"
set "ABAQUS_DIR=%ROOT%abaqus"
set "INP_DIR=%ROOT%abaqus\inp"
set "LOG_DIR=%ROOT%abaqus\logs"
set "ODB_DIR=%ROOT%abaqus\odb"
set "CSV_DIR=%ROOT%abaqus\csv"

echo.
echo ==========================================
echo Running Abaqus: %NAME%
echo ==========================================
echo CPU option: %CPU%
echo GPU option: %GPU%
if defined PRECISION_OPTION (echo Precision option: %PRECISION_OPTION%) else echo Precision option: Abaqus default (single precision)
echo.

if not exist "%INP_DIR%\%NAME%.inp" (
    echo ERROR: Input file not found: "%INP_DIR%\%NAME%.inp"
    exit /b 1
)

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%ODB_DIR%" mkdir "%ODB_DIR%"
if not exist "%CSV_DIR%" mkdir "%CSV_DIR%"

REM ==========================================
REM RUN ABAQUS
REM ==========================================

pushd "%INP_DIR%"
call abaqus job="%NAME%" input="%NAME%.inp" %PRECISION_OPTION%%ABAQUS_OPTIONS% interactive
set "ABAQUS_EXIT=%errorlevel%"
popd

if not "%ABAQUS_EXIT%"=="0" (
    echo.
    echo WARNING: Abaqus failed with exit code %ABAQUS_EXIT%.
    echo Continuing to collect any output files that were created.
) else (
    echo.
    echo Abaqus finished successfully.
)

REM ==========================================
REM MOVE ABAQUS OUTPUT
REM ==========================================

echo Moving STA...
if exist "%INP_DIR%\%NAME%.sta" (
    move /Y "%INP_DIR%\%NAME%.sta" "%LOG_DIR%\"
)

echo Moving DAT...
if exist "%INP_DIR%\%NAME%.dat" (
    move /Y "%INP_DIR%\%NAME%.dat" "%LOG_DIR%\"
)

echo Moving MSG...
if exist "%INP_DIR%\%NAME%.msg" (
    move /Y "%INP_DIR%\%NAME%.msg" "%LOG_DIR%\"
)

echo Moving ODB...
if exist "%INP_DIR%\%NAME%.odb" (
    move /Y "%INP_DIR%\%NAME%.odb" "%ODB_DIR%\"
)

if not exist "%ODB_DIR%\%NAME%.odb" (
    echo WARNING: ODB file not found. Skipping ODB extraction.
    goto cleanup
)

REM ==========================================
REM EXTRACT ODB
REM ==========================================

echo.
echo ==========================================
echo Extracting ODB
echo ==========================================
echo.

pushd "%ABAQUS_DIR%"

call abaqus python extract_odb.py "odb\%NAME%.odb"
set "EXTRACT_EXIT=%errorlevel%"

popd

if not "%EXTRACT_EXIT%"=="0" (
    echo.
    echo WARNING: ODB extraction failed with exit code %EXTRACT_EXIT%.
)

REM ==========================================
REM MOVE CSV
REM ==========================================

echo.
echo Moving CSV files...

if exist "%ODB_DIR%\%NAME%_field_U.csv" (
    move /Y "%ODB_DIR%\%NAME%_field_U.csv" "%CSV_DIR%\"
)

if exist "%ODB_DIR%\%NAME%_history.csv" (
    move /Y "%ODB_DIR%\%NAME%_history.csv" "%CSV_DIR%\"
)

REM ==========================================
REM CLEAN UP ABAQUS TEMPORARY FILES
REM ==========================================

:cleanup
echo.
echo Cleaning temporary Abaqus files...

for %%E in (abq com env mdl msg pac prt res sel stt) do (
    if exist "%INP_DIR%\%NAME%.%%E" del /Q "%INP_DIR%\%NAME%.%%E"
)

echo.
echo ==========================================
echo COMPLETE: %NAME%
echo ==========================================
echo.

if not "%ABAQUS_EXIT%"=="0" exit /b %ABAQUS_EXIT%
if defined EXTRACT_EXIT if not "%EXTRACT_EXIT%"=="0" exit /b %EXTRACT_EXIT%
exit /b 0
