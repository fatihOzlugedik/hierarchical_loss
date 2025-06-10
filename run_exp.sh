for alpha in 0 0.2 0.4 0.6 0.8 1 2; do
  for fold in {0..4}; do
    python run_pipeline_hiararchical.py --fold $fold --arch transformer --loss hl --alpha $alpha &
  done
  wait
done