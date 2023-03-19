#!/usr/bin/env bash
set -x

# source /mnt/lustre/share/spring/r0.3.0
ROOT=../../../
export PYTHONPATH=$ROOT:$PYTHONPATH

if [[ ! -d "logs" ]]; then
  mkdir logs
fi

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
################
gpus=${1-8}
TASK=${2-pose}
GINFO_INDEX=${3-0}   # task index config cherrypick (if necessary)
job_name=${4-debug}
################### ||| additional params usually not used
################### vvv
iter=${5-newest}
PRETRAIN_JOB_NAME=${6-${job_name}}
CONFIG=${7-${job_name}.yaml}
TEST_CONFIG=${8-vd_${TASK}_test.yaml}
TEST_MODEL=${9-checkpoints/${PRETRAIN_JOB_NAME}/ckpt_task${GINFO_INDEX}_iter_${iter}.pth.tar}
################

g=$((${gpus}<8?${gpus}:8))
echo 'start job:' ${job_name} ' config:' ${CONFIG} ' test_config:' ${TEST_CONFIG}


now=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=logs/${job_name}_test_${now}.log
echo 'log file: ' ${LOG_FILE}

GLOG_vmodule=MemcachedClient=-1 MKL_SERVICE_FORCE_INTEL=1 \
srun -n$1 -p <your partition> --debug --gres=gpu:${g} --ntasks-per-node=${g} --gpu  \
    --job-name=${job_name} --cpus-per-task=5 \
python -W ignore -u ${ROOT}/test.py \
    --expname ${job_name} \
    --config ${CONFIG} \
    --test_config ${TEST_CONFIG} \
    --spec_ginfo_index ${GINFO_INDEX} \
    --load-path=${TEST_MODEL} \
    --tcp_port $PORT \
    2>&1 | tee ${LOG_FILE}
