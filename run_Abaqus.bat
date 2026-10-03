@echo off
setlocal

if "%~1"=="" (
    echo.
    echo Usage: run_Abaqus.bat MODEL_NAME
    echo.
    exit /b 1
)

set "NAME=%~1"
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
call abaqus job="%NAME%" input="%NAME%.inp" double=both interactive
set "ABAQUS_EXIT=%errorlevel%"
popd

if not "%ABAQUS_EXIT%"=="0" (
    echo.
    echo ERROR: Abaqus failed.
    pause
    exit /b 1
)

echo.
echo Abaqus finished successfully.
echo.

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

echo Moving ODB...
if exist "%INP_DIR%\%NAME%.odb" (
    move /Y "%INP_DIR%\%NAME%.odb" "%ODB_DIR%\"
)

if not exist "%ODB_DIR%\%NAME%.odb" (
    echo ERROR: ODB file not found: "%ODB_DIR%\%NAME%.odb"
    exit /b 1
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
    echo ERROR: ODB extraction failed.
    pause
    exit /b 1
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

echo.
echo Cleaning temporary Abaqus files...

for %%E in (abq com mdl msg pac prt res sel stt) do (
    if exist "%INP_DIR%\%NAME%.%%E" del /Q "%INP_DIR%\%NAME%.%%E"
)

echo.
echo ==========================================
echo COMPLETE: %NAME%
echo ==========================================
echo.
