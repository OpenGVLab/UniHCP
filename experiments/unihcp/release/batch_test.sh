#!/usr/bin/env bash
#set -x

# source /mnt/lustre/share/spring/r0.3.0
ROOT=../../../
export PYTHONPATH=$ROOT:$PYTHONPATH

if [[ ! -d "logs" ]]; then
  mkdir logs
fi

################
gpus=${1-8}

job_name=${2-debug}
################### ||| additional params usually not used
################### vvv
iter=${3-newest}
PRETRAIN_JOB_NAME=${4-${job_name}}
CONFIG=${5-${job_name}.yaml}
################
g=$((${gpus}<8?${gpus}:8))

#### test list
declare -A test_info_list
test_info_list[pose_lpe]=0
test_info_list[ochuman_pose]=0
#test_info_list[pose_mpii_lpe]=x  # requires mpii queries
test_info_list[par_lpe]=1
#test_info_list[par_atr_lpe]=x  # requires ATR queries
test_info_list[reid]=2
test_info_list[reid_cuhk3]=2
test_info_list[reid_duke]=2
test_info_list[reid_msmt]=2
test_info_list[reid_senseid]=2
test_info_list[par_lip_lpe]=3
test_info_list[par_cihp_lpe]=4
test_info_list[pa100k_lpe]=5
test_info_list[rap2_lpe]=6
#test_info_list[peta_lpe]=x  # requires PETA queries
test_info_list[pose_aic_lpe]=7
test_info_list[peddet_caltech]=8
test_info_list[peddet_inter_lpe]=8

for TASK in "${!test_info_list[@]}"; do
  full_job_name=${job_name}_test_${TASK}
  now=$(date +"%Y%m%d_%H%M%S")
  GINFO_INDEX=${test_info_list[${TASK}]}
  LOG_FILE=logs/${full_job_name}_test_${TASK}_${now}.log
  echo "=======>${TASK} log file: ${LOG_FILE}"
  TEST_CONFIG=vd_${TASK}_test.yaml
  TEST_MODEL=checkpoints/${PRETRAIN_JOB_NAME}/ckpt_task${GINFO_INDEX}_iter_${iter}.pth.tar
  echo 'start job:' ${full_job_name} ' config:' ${CONFIG} ' test_config:' ${TEST_CONFIG}

  while true # find unused tcp port
  do
      PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
      status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
      if [ "${status}" != "0" ]; then
          break;
      fi
  done

  GLOG_vmodule=MemcachedClient=-1 MKL_SERVICE_FORCE_INTEL=1 \
  srun -n$1 -p <your partition>  --gres=gpu:${g} --ntasks-per-node=${g} --gpu  \
      --job-name=${full_job_name} --cpus-per-task=5 \
  python -W ignore -u ${ROOT}/test.py \
      --expname ${full_job_name} \
      --config ${CONFIG} \
      --test_config ${TEST_CONFIG} \
      --spec_ginfo_index ${GINFO_INDEX} \
      --load-path=${TEST_MODEL} \
      --tcp_port $PORT \
      2>&1 | tee ${LOG_FILE} &

  sleep 10
done


