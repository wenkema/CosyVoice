#!/bin/bash

SH_DIR=$(cd "$(dirname "$0")"; pwd)
PRJ_DIR=$(dirname $SH_DIR)

# install ttsfrd
# TODO: support downloading
if [ -d "${PRJ_DIR}/pretrained_models/CosyVoice-ttsfrd" ]; then
    echo "install ttsfrd"
    cd "${PRJ_DIR}/pretrained_models/CosyVoice-ttsfrd"
    pip install ttsfrd_dependency-0.1-py3-none-any.whl
    pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
fi
