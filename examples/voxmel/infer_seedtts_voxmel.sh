#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${workspace}/../.." && pwd)"

PY="${PY:-python3}"
QWEN_PATH="${QWEN_PATH:-${workspace}/qwen_tokenizer}"
VOCOS_PATH="${VOCOS_PATH:-/path/vocos-mel-24khz}"

# exp_dir="${EXP_DIR:-${workspace}/exps/large}"
# ckpt="${CKPT:-model_L_final.pt}"
exp_dir="${EXP_DIR:-${workspace}/exps/medium}"
ckpt="${CKPT:-model_M_final.pt}"
testset="${TESTSET:-/path/seedtts_testset}"
langs="${LANGS:-zh,en}"
lst_name="${LST_NAME:-meta.lst}"
output_dir="${OUTPUT_DIR:-${exp_dir}/output/seedtts_${ckpt%.pt}}"

device="${DEVICE:-cuda}"
GPU="${GPU:-0}"
NPROC="${NPROC:-1}"
MAX_ITEMS="${MAX_ITEMS:-0}"
CFG_TEXT="${CFG_TEXT:-3.5}"
CFG_SPK="${CFG_SPK:-3}"
UP_STEPS="${UP_STEPS:-1}"
COARSE_STEPS="${COARSE_STEPS:-32}"

extra_args=("++up_num_steps=${UP_STEPS}" "++coarse_steps=${COARSE_STEPS}")
extra_args+=("++cfg_text_strength=${CFG_TEXT}" "++cfg_spk_strength=${CFG_SPK}")
[[ -n "${MIN_SECONDS:-}" ]] && extra_args+=("++min_seconds=${MIN_SECONDS}")
[[ -n "${MAX_SECONDS:-}" ]] && extra_args+=("++max_seconds=${MAX_SECONDS}")

if [[ ! "${NPROC}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC must be a positive integer, got: ${NPROC}" >&2
    exit 1
fi
for path in "${exp_dir}/config.yaml" "${exp_dir}/${ckpt}"; do
    if [[ ! -f "${path}" ]]; then
        echo "required file not found: ${path}" >&2
        exit 1
    fi
done

mkdir -p "${output_dir}"
echo "exp_dir : ${exp_dir}"
echo "ckpt    : ${ckpt}"
echo "testset : ${testset} (${langs}, ${lst_name})"
echo "output  : ${output_dir}"
echo "parallel: GPU=${GPU} NPROC=${NPROC}"

cd "${workspace}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

IFS=',' read -ra gpu_arr <<< "${GPU}"
n_gpus=${#gpu_arr[@]}
total=$(( NPROC * n_gpus ))
pids=()
for (( sid=0; sid<total; sid++ )); do
    CUDA_VISIBLE_DEVICES="${gpu_arr[$(( sid % n_gpus ))]}" "${PY}" infer_voxmel_seedtts.py \
        --config-path="${exp_dir}" \
        --config-name="config.yaml" \
        ++init_param="${exp_dir}/${ckpt}" \
        ++excludes="'mel_mean,mel_std'" \
        ++qwen_pretrain_path="${QWEN_PATH}" \
        ++model_conf.load_checkpoint=false \
        ++model_conf.cfg_batch=true \
        ++vocos_path="${VOCOS_PATH}" \
        ++use_ema=false \
        +input="${testset}" \
        ++langs="'${langs}'" \
        ++lst_name="${lst_name}" \
        ++output_dir="${output_dir}" \
        ++device="${device}" \
        ++max_items="${MAX_ITEMS}" \
        +num_shards="${total}" +shard_id="${sid}" "${extra_args[@]}" &
    pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do
    wait "${pid}" || rc=$?
done
exit ${rc}
