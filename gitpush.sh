#!/bin/bash

# Get the commit message from the first argument
COMMIT_MSG="$1"

echo "--- [$(date)] Starting script ---"

# 1. Check if a commit message was provided
if [ -z "$COMMIT_MSG" ]; then
    echo "ERROR: No commit message provided."
    echo "Usage: ./push.sh \"Your commit message\""
    exit 1 # Exit with an error
fi

echo "--- 2. Adding all files to staging ---"
git add .

echo "--- 3. Committing files ---"
# We use "$COMMIT_MSG" in quotes to handle messages with spaces
git commit -am "$COMMIT_MSG"

echo "--- 4. Pushing to origin main ---"
git push origin main

echo "--- All done. ---"

