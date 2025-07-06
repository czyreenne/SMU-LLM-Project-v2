#!/bin/bash

echo "Legal Analysis Simulation System"
echo "================================"

# PARSE COMMAND LINE ARGUMENTS. EDIT HERE 
MODEL=""
QUESTION=""
HYPOTHETICAL=""

# UNUSED! flag for potential Q&A based on result
INTERACTIVE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --interactive)
      INTERACTIVE="--interactive"
      shift
      ;;
    --question)
      if [ ! -z "$HYPOTHETICAL" ]; then
        echo "Error: Cannot provide both --question and --hypo flags. Please choose one."
        exit 1
      fi
      QUESTION="$2"
      shift 2
      ;;
    --hypo)
      if [ ! -z "$QUESTION" ]; then
        echo "Error: Cannot provide both --question and --hypo flags. Please choose one."
        exit 1
      fi
      HYPOTHETICAL="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      shift
      ;;
  esac
done

# If model not provided via command line, prompt the user
if [ -z "$MODEL" ]; then
  echo
  echo "Select a model (or press Enter to run all models):"
  echo "1. gpt-4o-mini"
  echo "2. gpt-4o"
  echo "3. claude-3-opus"
  echo "4. claude-3-sonnet"
  echo "5. deepseek-chat"
  echo "6. Run all models"
  echo
  read -p "Enter your choice (1-6, or press Enter for all models, or enter a custom model name): " MODEL_CHOICE
  
  case $MODEL_CHOICE in
    1) MODEL="gpt-4o-mini" ;;
    2) MODEL="gpt-4o" ;;
    3) MODEL="claude-3-opus" ;;
    4) MODEL="claude-3-sonnet" ;;
    5) MODEL="deepseek-chat" ;;
    6) MODEL="" ;;
    "") MODEL="" ;;
    [1-6]) ;; # Do nothing for valid numbers already handled
    *)
      # If not a number 1-6, assume it's a custom model name
      if [ ! -z "$MODEL_CHOICE" ]; then
        MODEL="$MODEL_CHOICE"
      fi
      ;;
  esac
fi

# If neither question nor hypothetical provided via command line, prompt for one
if [ -z "$QUESTION" ] && [ -z "$HYPOTHETICAL" ]; then
  echo
  echo "Would you like to provide a legal question or a hypothetical directory?"
  echo "1. Legal Question"
  echo "2. Hypothetical Directory"
  read -p "Enter your choice (1 or 2): " INPUT_CHOICE
  
  case $INPUT_CHOICE in
    1)
      read -p "Enter your legal question: " QUESTION
      ;;
    2)
      read -p "Enter the path to your hypothetical directory: " HYPOTHETICAL
      # Verify the directory exists
      if [ ! -d "$HYPOTHETICAL" ]; then
        echo "Error: The specified directory does not exist or is not a directory."
        exit 1
      fi
      ;;
    *)
      echo "Invalid choice. Defaulting to legal question."
      read -p "Enter your legal question: " QUESTION
      ;;
  esac
fi

# Build the command
CMD="python3 main.py"

# Add model if specified
if [ ! -z "$MODEL" ]; then
  CMD="$CMD --model \"$MODEL\""
fi

# Add interactive flag if specified
if [ ! -z "$INTERACTIVE" ]; then
  CMD="$CMD $INTERACTIVE"
fi

# Add question if specified
if [ ! -z "$QUESTION" ]; then
  CMD="$CMD --question \"$QUESTION\""
fi

# Add hypothetical if specified
if [ ! -z "$HYPOTHETICAL" ]; then
  CMD="$CMD --hypo \"$HYPOTHETICAL\""
fi

echo
echo "Running command: $CMD"
echo

eval $CMD

echo
echo "Analysis completed."