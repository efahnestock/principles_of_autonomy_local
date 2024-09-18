#!/bin/bash

# Check if exactly one argument is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <HW_TAG> (e.g., ps2)"
  exit 1
fi

# exit if build fails
set -e 

docker compose build --build-arg HW_TAG=$1
docker compose up