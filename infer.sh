PROMPT='A beautiful scenery of mountains during sunrise, with vibrant colors and a serene atmosphere.'

IMAGE_PATH="/workspace/HunyuanVideo-1.5/input/222851.png" # Optional, none or <image path> to enable i2v mode
SEED=34344945
python generate.py \
  --prompt "$PROMPT" \
  --image_path $IMAGE_PATH \
  --resolution 480p \
  --model_path ckpts \
  --num_inference_steps 20 \
  --cfg_distilled true \
  --sparse_attn true \
  --seed $SEED \
  --sr false 