#!/usr/bin/env bash
# =============================================================================
#  V4_LARGE+2 实验编排
#  VLM  = Qwen/Qwen3-VL-8B-Instruct  (hidden=4096, vocab=151936)
#  DEC  = Qwen/Qwen3-4B               (hidden=2560, 36L, vocab=151936)
# =============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"

OUT_ROOT="$PROJECT_DIR/output"
EXP_NAME="v4_large_Qwen3-VL-8B_Qwen3-4B"
EXP_ROOT="$OUT_ROOT/$EXP_NAME"
LOG_DIR="$EXP_ROOT/logs"
mkdir -p "$LOG_DIR"
RUN_TAG=$(date +%Y%m%d_%H%M%S)
exec > >(tee -a "$LOG_DIR/pipeline_${RUN_TAG}.log") 2>&1

PY=${PY:-/raid/yewei/miniconda3/envs/llm/bin/python}
ROOT=$EXP_ROOT
SPLITS=$OUT_ROOT/splits_general.json   # 全实验共享 splits
EVAL_DIR=$ROOT/eval_v4
CKPT_DIR=$ROOT/ckpt
mkdir -p $EVAL_DIR $CKPT_DIR

VLM=Qwen/Qwen3-VL-8B-Instruct
DEC=Qwen/Qwen3-4B
LMBDA=0.3
EPOCHS=3
BATCH=8                               # 全实验统一 effective batch=8
LR=5e-4

# ---------- Step 0: splits ----------
if [[ ! -f $SPLITS ]]; then
  $PY -m zoom_decoder.make_splits \
    --out_file $SPLITS \
    --splits perception reasoning \
    --seed 0 \
    --train_per_cell 70 --val_per_cell 30
fi

# ---------- Step 1: teacher dstores (VLM=Qwen3-VL-8B, DEC=Qwen3-4B) ----------
for mode in zoom priv zoom_priv; do
  if [[ "$mode" == "zoom_priv" ]]; then
    out=$ROOT/dstore_zoompriv_perc
  else
    out=$ROOT/dstore_${mode}_perc
  fi
  if [[ ! -d $out/dataset ]]; then
    $PY -m zoom_decoder.prepare_teacher \
      --vlm_model $VLM \
      --splits perception \
      --splits_file $SPLITS \
      --teacher_mode $mode \
      --out_dir $out \
      --topk 32 --seed 0
  fi
done

if [[ ! -d $ROOT/dstore_priv_all/dataset ]]; then
  $PY -m zoom_decoder.prepare_teacher \
    --vlm_model $VLM \
    --splits perception reasoning \
    --splits_file $SPLITS \
    --teacher_mode priv \
    --out_dir $ROOT/dstore_priv_all \
    --topk 32 --seed 0
fi

if [[ ! -d $ROOT/dstore_knn_all/dataset ]]; then
  $PY -m zoom_decoder.prepare_knn_teacher \
    --vlm_model $VLM \
    --splits perception reasoning \
    --splits_file $SPLITS \
    --out_dir $ROOT/dstore_knn_all \
    --topk 32 --knn_k 64 --knn_temp 500.0 --seed 0
fi

# perception-only kNN teacher dstore (decoupling control)
if [[ ! -d $ROOT/dstore_knn_perc/dataset ]]; then
  $PY -m zoom_decoder.prepare_knn_teacher \
    --vlm_model $VLM \
    --splits perception \
    --splits_file $SPLITS \
    --out_dir $ROOT/dstore_knn_perc \
    --topk 32 --knn_k 64 --knn_temp 500.0 --seed 0
fi

# ---------- Step 2: 训练 ----------
train_zd() {
  local name=$1 ddir=$2 nlay=$3 extra=${4:-}
  if [[ ! -d $CKPT_DIR/$name/final ]]; then
    $PY -m zoom_decoder.train \
      --data_dir $ddir \
      --decoder_model $DEC \
      --tokenizer_model $VLM \
      --num_layers $nlay \
      --out_dir $CKPT_DIR/$name \
      --epochs $EPOCHS --batch_size $BATCH --lr $LR \
      $extra
  fi
}

# with-zoom group
train_zd zd_zoom_6L_perc        $ROOT/dstore_zoom_perc        6
train_zd zd_zoom_24L_perc       $ROOT/dstore_zoom_perc       24
train_zd zd_priv_24L_perc       $ROOT/dstore_priv_perc       24
train_zd zd_zoompriv_24L_perc   $ROOT/dstore_zoompriv_perc  24
train_zd zd_zoom_24L_noap_perc  $ROOT/dstore_zoom_perc       24 "--disable_aperture"
train_zd zd_zoom_24L_nosw_perc  $ROOT/dstore_zoom_perc       24 "--disable_scale_weight"
train_zd zd_zoom_24L_plain_perc $ROOT/dstore_zoom_perc       24 "--disable_aperture --disable_scale_weight"
# decoupling control
train_zd zd_knn_24L_perc        $ROOT/dstore_knn_perc        24
# Qwen3-4B full-depth (36L) scaling probe
train_zd zd_zoom_36L_perc       $ROOT/dstore_zoom_perc       36

# without-zoom group
train_zd zd_priv_24L_all      $ROOT/dstore_priv_all        24
train_zd zd_knn_6L_all        $ROOT/dstore_knn_all          6
train_zd zd_knn_6L_noap_all   $ROOT/dstore_knn_all          6 "--disable_aperture"
train_zd zd_knn_6L_nosw_all   $ROOT/dstore_knn_all          6 "--disable_scale_weight"
train_zd zd_knn_6L_plain_all  $ROOT/dstore_knn_all          6 "--disable_aperture --disable_scale_weight"
train_zd zd_knn_24L_all       $ROOT/dstore_knn_all         24
train_zd zd_knn_24L_noap_all  $ROOT/dstore_knn_all         24 "--disable_aperture"
train_zd zd_knn_24L_nosw_all  $ROOT/dstore_knn_all         24 "--disable_scale_weight"
train_zd zd_knn_24L_plain_all $ROOT/dstore_knn_all         24 "--disable_aperture --disable_scale_weight"
# Qwen3-4B full-depth (36L) scaling probe
train_zd zd_knn_36L_all       $ROOT/dstore_knn_all         36

# LoRA r=16
if [[ ! -d $CKPT_DIR/lora_r16_all/final ]]; then
  $PY -m zoom_decoder.train_lora \
    --vlm_model $VLM \
    --splits_file $SPLITS \
    --train_splits perception reasoning \
    --out_dir $CKPT_DIR/lora_r16_all \
    --epochs 3 --batch_size 1 --grad_accum 2 --lr 1e-4 --lora_rank 16
fi

# ---------- Step 3: 评测 ----------
eval_zoom() {
  local name=$1
  $PY -m zoom_decoder.evaluate --mode zoom \
    --vlm_model $VLM \
    --zoom_ckpt $CKPT_DIR/$name/final \
    --splits_file $SPLITS \
    --eval_splits perception reasoning \
    --lmbda $LMBDA --max_new_tokens 512 \
    --out_file $EVAL_DIR/$name.json
}

$PY -m zoom_decoder.evaluate --mode base \
  --vlm_model $VLM \
  --splits_file $SPLITS \
  --eval_splits perception reasoning \
  --max_new_tokens 512 \
  --out_file $EVAL_DIR/base.json

for name in \
  zd_zoom_6L_perc \
  zd_zoom_24L_perc zd_priv_24L_perc zd_zoompriv_24L_perc \
  zd_zoom_24L_noap_perc zd_zoom_24L_nosw_perc zd_zoom_24L_plain_perc \
  zd_knn_24L_perc \
  zd_zoom_36L_perc \
  zd_priv_24L_all \
  zd_knn_6L_all zd_knn_6L_noap_all zd_knn_6L_nosw_all zd_knn_6L_plain_all \
  zd_knn_24L_all zd_knn_24L_noap_all zd_knn_24L_nosw_all zd_knn_24L_plain_all \
  zd_knn_36L_all
do
  eval_zoom $name
done

$PY -m zoom_decoder.evaluate --mode lora \
  --vlm_model $VLM \
  --lora_ckpt $CKPT_DIR/lora_r16_all/final \
  --splits_file $SPLITS \
  --eval_splits perception reasoning \
  --max_new_tokens 512 \
  --out_file $EVAL_DIR/lora_r16_all.json

echo "==== V4_LARGE+2 全实验完成，结果在 $EVAL_DIR/ ===="

# ---------- Step 4: 生成报告 ----------
$PY -m zoom_decoder.summarize_v4 \
  --eval_dir $EVAL_DIR \
  --out_file $ROOT/V4_RESULTS.md \
  --splits_file $SPLITS
echo "报告已写入 $ROOT/V4_RESULTS.md"

cp -f "$ROOT/V4_RESULTS.md" "$ROOT/README.md"
echo "README 报告已写入 $ROOT/README.md"

$PY scripts/plot_loss_compare.py \
  --output_root "$OUT_ROOT" \
  --out_file "$EXP_ROOT/loss_compare.png" || true
