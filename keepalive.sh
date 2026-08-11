#!/bin/bash
while true; do
  curl -s https://signal-project.onrender.com/health > /dev/null
  echo "Pinged health endpoint at $(date)"
  sleep 540
done
