@echo off
setlocal EnableDelayedExpansion

echo Legal Analysis Simulation System
echo ================================

REM PARSE COMMAND LINE ARGUMENTS
set MODEL=
set QUESTION=
set HYPOTHETICAL=input

REM UNUSED! flag for potential Q&A based on result
set INTERACTIVE=

:parse_args
if "%~1"=="" goto end_parse_args
if /i "%~1"=="--model" (
    set MODEL=%~2
    shift
) else if /i "%~1"=="--interactive" (
    set INTERACTIVE=--interactive
) else if /i "%~1"=="--question" (
    if not "!HYPOTHETICAL!"=="" (
        echo Error: Cannot provide both --question and --hypo flags. Please choose one.
        exit /b 1
    )
    set QUESTION=%~2
    shift
) else if /i "%~1"=="--hypo" (
    if not "!QUESTION!"=="" (
        echo Error: Cannot provide both --question and --hypo flags. Please choose one.
        exit /b 1
    )
    set HYPOTHETICAL=%~2
    shift
)
shift
goto parse_args
:end_parse_args

REM If model not provided via command line, prompt the user
if "!MODEL!"=="" (
    echo.
    echo Select a model (or press Enter to run all models):
    echo 1. gpt-4o-mini
    echo 2. gpt-4o
    echo 3. claude-3-opus
    echo 4. claude-3-sonnet
    echo 5. deepseek-chat
    echo 6. Run all models
    echo.
    set /p MODEL_CHOICE="Enter your choice (1-6, or press Enter for all models, or enter a custom model name): "
    
    if "!MODEL_CHOICE!"=="1" set MODEL=gpt-4o-mini
    if "!MODEL_CHOICE!"=="2" set MODEL=gpt-4o
    if "!MODEL_CHOICE!"=="3" set MODEL=claude-3-opus
    if "!MODEL_CHOICE!"=="4" set MODEL=claude-3-sonnet
    if "!MODEL_CHOICE!"=="5" set MODEL=deepseek-chat
    if "!MODEL_CHOICE!"=="6" set MODEL=
    if "!MODEL_CHOICE!"=="" set MODEL=
    
    REM If not a number 1-6 and not empty, assume it's a custom model name
    echo !MODEL_CHOICE! | findstr /r "^[1-6]$" >nul
    if errorlevel 1 if not "!MODEL_CHOICE!"=="" set MODEL=!MODEL_CHOICE!
)

REM If neither question nor hypothetical provided, prompt for one
if "!QUESTION!"=="" if "!HYPOTHETICAL!"=="" (
    echo.
    echo Would you like to provide a legal question or a hypothetical directory?
    echo 1. Legal Question
    echo 2. Hypothetical Directory
    echo.
    set /p INPUT_CHOICE="Enter your choice (1 or 2): "
    
    if "!INPUT_CHOICE!"=="1" (
        set /p QUESTION="Enter your legal question: "
    ) else if "!INPUT_CHOICE!"=="2" (
        set /p HYPOTHETICAL="Enter the path to your hypothetical directory: "
        REM Verify the directory exists
        if not exist "!HYPOTHETICAL!\" (
            echo Error: The specified directory does not exist or is not a directory.
            exit /b 1
        )
    ) else (
        echo Invalid choice. Defaulting to legal question.
        set /p QUESTION="Enter your legal question: "
    )
)

REM Build the command
set CMD=python main.py

REM Add model if specified
if not "!MODEL!"=="" (
    set CMD=!CMD! --model "!MODEL!"
)

REM Add interactive flag if specified
if not "!INTERACTIVE!"=="" (
    set CMD=!CMD! !INTERACTIVE!
)

REM Add question if specified
if not "!QUESTION!"=="" (
    set CMD=!CMD! --question "!QUESTION!"
)

REM Add hypothetical if specified
if not "!HYPOTHETICAL!"=="" (
    set CMD=!CMD! --hypo "!HYPOTHETICAL!"
)

echo.
echo Running command: !CMD!
echo.

REM Execute the command
!CMD!

REM Pause at the end so results can be viewed
echo.
echo Analysis completed. Press any key to exit...
pause >nul
endlocal