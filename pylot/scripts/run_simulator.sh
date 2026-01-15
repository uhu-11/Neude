#!/bin/bash

if [ -z "$CARLA_HOME" ]; then
    echo "Please set \$CARLA_HOME before running this script"
    exit 1
fi

if [ -z "$1" ]; then
    PORT=2000
else
    PORT=$1
fi

SDL_VIDEODRIVER=offscreen ${CARLA_HOME}/CarlaUE4.sh -opengl -windowed -ResX=1920 -ResY=1080 -carla-server -world-port=$PORT -benchmark -fps=20 -quality-level=Low
# ${CARLA_HOME}/CarlaUE4.sh -opengl -windowed -ResX=1920 -ResY=1080 -carla-server -world-port=$PORT -benchmark -fps=20 -quality-level=Low
