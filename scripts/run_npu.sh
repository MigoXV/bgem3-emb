#!/usr/bin/env bash
# BGE 使用独立的 Torch 2.8 / CANN 8.3 环境，避免继承 LLM 的 CANN 9.1。
set -eo pipefail
cd /root/repositories/bgem3-emb
unset VIRTUAL_ENV PYTHONPATH ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH ASCEND_AICPU_PATH
export PATH=/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export BGE_MODEL_DIR="${BGE_MODEL_DIR:-/home/model-bin/BAAI/bge-m3}"
export BGE_DEVICE="${BGE_DEVICE:-npu:0}"
export BGE_DTYPE="${BGE_DTYPE:-float16}"
exec poetry run bgem3-emb serve "$@"
