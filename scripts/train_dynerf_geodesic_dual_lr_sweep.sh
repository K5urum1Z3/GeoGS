#!/bin/bash
set -euo pipefail

scene_list=("cut_roasted_beef" "flame_steak" "sear_steak")

GPU=0
PORT=7489
DATA_ROOT="/home/junjie/dataset/dynerf"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GEO_LATENT_DIMS=(128)                 
GEO_STEP_SIZES=(0.01)                 
GEO_LR_SPHERES=(0.000025)             
GEO_LR_HYPERS=(0.000025)      

GEO_STEP_SIZE_SPHERES=(0.0025 0.005)  
GEO_STEP_SIZE_HYPERS=(0.0025 0.005)   

GEO_LR_SPHERE_FINALS=(0.0000025 0.000005) 
GEO_LR_HYPER_FINALS=(0.0000025 0.000005)  
GEO_LR_DELAY_MULTS=(0.01 0.05)       

# 去重策略：
# 当 sphere/hyper 分支步长显式设置为正数时，geo_step_size 仅作为后备值，不再做独立遍历
# 0 -> 不遍历 GEO_STEP_SIZES（推荐，避免重复组合）
# 1 -> 仍遍历 GEO_STEP_SIZES
SWEEP_BASE_STEP_WITH_BRANCH_STEPS=0
BASE_GEO_STEP_FALLBACK=0.01

# 断点续跑策略：
# 0 -> 只补跑“没有日志文件”的组合（当前默认，满足补跑剩余29组）
# 1 -> 对已有但未完成的日志也重跑
RERUN_INCOMPLETE_LOGS=0

log_dir="${REPO_ROOT}/logs/geodesic_dual_lr_sweep"
mkdir -p "$log_dir"

is_log_complete() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  grep -q "Training complete\\." "$f"
}

if [[ ${SWEEP_BASE_STEP_WITH_BRANCH_STEPS} -eq 1 ]]; then
  EFFECTIVE_GEO_STEP_SIZES=("${GEO_STEP_SIZES[@]}")
else
  EFFECTIVE_GEO_STEP_SIZES=("${BASE_GEO_STEP_FALLBACK}")
fi

scene_combo_total=$(( ${#GEO_LATENT_DIMS[@]} * ${#EFFECTIVE_GEO_STEP_SIZES[@]} * ${#GEO_LR_SPHERES[@]} * ${#GEO_LR_HYPERS[@]} * ${#GEO_STEP_SIZE_SPHERES[@]} * ${#GEO_STEP_SIZE_HYPERS[@]} * ${#GEO_LR_SPHERE_FINALS[@]} * ${#GEO_LR_HYPER_FINALS[@]} * ${#GEO_LR_DELAY_MULTS[@]} ))
total_combo_all=$(( scene_combo_total * ${#scene_list[@]} ))
echo "Sweep total combinations: ${total_combo_all} (${#scene_list[@]} scene x ${scene_combo_total})"
echo "Resume mode RERUN_INCOMPLETE_LOGS=${RERUN_INCOMPLETE_LOGS}"
echo "SWEEP_BASE_STEP_WITH_BRANCH_STEPS=${SWEEP_BASE_STEP_WITH_BRANCH_STEPS} (effective_geo_step_count=${#EFFECTIVE_GEO_STEP_SIZES[@]})"

overall_total=0
overall_skip_done=0
overall_skip_existing=0
overall_run=0
overall_fail=0

for scene in "${scene_list[@]}"; do
  echo "========================================="
  echo "Start sweep scene: $scene"
  echo "========================================="
  if [[ ! -d "${DATA_ROOT}/${scene}" ]]; then
    echo "[Error] Scene path not found: ${DATA_ROOT}/${scene}"
    echo "Please check DATA_ROOT in script: ${DATA_ROOT}"
    exit 1
  fi

  scene_total=0
  scene_skip_done=0
  scene_skip_existing=0
  scene_run=0
  scene_fail=0

  for geo_latent_dim in "${GEO_LATENT_DIMS[@]}"; do
    for geo_step_size in "${EFFECTIVE_GEO_STEP_SIZES[@]}"; do
      for geo_lr_sphere in "${GEO_LR_SPHERES[@]}"; do
        for geo_lr_hyper in "${GEO_LR_HYPERS[@]}"; do
          for geo_step_size_sphere in "${GEO_STEP_SIZE_SPHERES[@]}"; do
            for geo_step_size_hyper in "${GEO_STEP_SIZE_HYPERS[@]}"; do
              for geo_lr_sphere_final in "${GEO_LR_SPHERE_FINALS[@]}"; do
                for geo_lr_hyper_final in "${GEO_LR_HYPER_FINALS[@]}"; do
                  for geo_lr_delay_mult in "${GEO_LR_DELAY_MULTS[@]}"; do
                    scene_total=$((scene_total + 1))
                    overall_total=$((overall_total + 1))

                    tag="ld${geo_latent_dim}_step${geo_step_size}_lrs${geo_lr_sphere}_lrh${geo_lr_hyper}_steps${geo_step_size_sphere}_steph${geo_step_size_hyper}_lrsf${geo_lr_sphere_final}_lrhf${geo_lr_hyper_final}_gdm${geo_lr_delay_mult}"
                    expname="dynerf/${scene}_geo_sphere_hyper_${tag}"
                    log_file="${log_dir}/${scene}_${tag}.txt"

                    if is_log_complete "${log_file}"; then
                      echo "[Skip][Done] ${scene}_${tag}"
                      scene_skip_done=$((scene_skip_done + 1))
                      overall_skip_done=$((overall_skip_done + 1))
                      continue
                    fi

                    if [[ ${RERUN_INCOMPLETE_LOGS} -eq 0 && -f "${log_file}" ]]; then
                      echo "[Skip][Existing-Incomplete] ${scene}_${tag}"
                      scene_skip_existing=$((scene_skip_existing + 1))
                      overall_skip_existing=$((overall_skip_existing + 1))
                      continue
                    fi

                    if [[ ${RERUN_INCOMPLETE_LOGS} -eq 1 && -f "${log_file}" ]]; then
                      backup_log="${log_file%.txt}.incomplete_$(date +%Y%m%d_%H%M%S).txt"
                      mv "${log_file}" "${backup_log}"
                      echo "[Info] Backup old incomplete log: ${backup_log}"
                    fi

                    scene_run=$((scene_run + 1))
                    overall_run=$((overall_run + 1))

                    echo "-----------------------------------------"
                    echo "Scene: ${scene}"
                    echo "geo_latent_dim: ${geo_latent_dim}"
                    echo "geo_step_size(base/sphere/hyper): ${geo_step_size} / ${geo_step_size_sphere} / ${geo_step_size_hyper}"
                    echo "geo_lr_sphere/hyper: ${geo_lr_sphere} / ${geo_lr_hyper}"
                    echo "geo_lr_sphere_final/hyper_final: ${geo_lr_sphere_final} / ${geo_lr_hyper_final}"
                    echo "geo_lr_delay_mult: ${geo_lr_delay_mult}"
                    echo "Exp: ${expname}"
                    echo "Log: ${log_file}"
                    echo "-----------------------------------------"

                    set +e
                    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
                      -s "${DATA_ROOT}/${scene}" \
                      --port "${PORT}" \
                      --expname "${expname}" \
                      --configs "arguments/dynerf/${scene}.py" \
                      --use_geodesic_branch \
                      --use_sphere_branch \
                      --use_hyper_branch \
                      --geo_latent_dim "${geo_latent_dim}" \
                      --geo_step_size "${geo_step_size}" \
                      --geo_step_size_sphere "${geo_step_size_sphere}" \
                      --geo_step_size_hyper "${geo_step_size_hyper}" \
                      --geo_lr_sphere "${geo_lr_sphere}" \
                      --geo_lr_hyper "${geo_lr_hyper}" \
                      --geo_lr_sphere_final "${geo_lr_sphere_final}" \
                      --geo_lr_hyper_final "${geo_lr_hyper_final}" \
                      --geo_lr_delay_mult "${geo_lr_delay_mult}" \
                      2>&1 | tee "${log_file}"
                    run_status=${PIPESTATUS[0]}
                    set -e

                    if [[ ${run_status} -ne 0 ]]; then
                      echo "[Failed] ${scene}_${tag} (exit=${run_status})"
                      scene_fail=$((scene_fail + 1))
                      overall_fail=$((overall_fail + 1))
                    else
                      echo "[Done] ${scene}_${tag}"
                    fi
                  done
                done
              done
            done
          done
        done
      done
    done
  done

  echo "Finished sweep scene: $scene"
  echo "Scene summary: total=${scene_total}, skip_done=${scene_skip_done}, skip_existing_incomplete=${scene_skip_existing}, run=${scene_run}, failed=${scene_fail}"
done

echo "========================================="
echo "All sweeps finished!"
echo "Summary: total=${overall_total}, skip_done=${overall_skip_done}, skip_existing_incomplete=${overall_skip_existing}, run=${overall_run}, failed=${overall_fail}"
echo "========================================="
