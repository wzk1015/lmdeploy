set -x
# partition='VC3'
partition='INTERN3'
# TYPE='reserved'
TYPE='spot'
JOB_NAME='lmdeploy_inference'
GPUS=1
GPUS_PER_NODE=1
# GPUS=8
# GPUS_PER_NODE=8
CPUS_PER_TASK=12

# export TORCH_USE_CUDA_DSA=1

# SRUN_ARGS="--jobid=3674764"

srun -p $partition --job-name=${JOB_NAME} \
  --gres=gpu:${GPUS_PER_NODE} --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} -n${GPUS} \
  --quotatype=${TYPE} --kill-on-bad-exit=1 \
  ${SRUN_ARGS} \
python -u lmdeploy_test.py
