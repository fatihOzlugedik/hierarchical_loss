for arch in transformer wbcmil; do
  for fold in {0..4}; do
    python /lustre/groups/labs/marr/qscd01/workspace/fatih.oezluegedik/hierarchical_loss/run_pipeline_hiararchical.py --fold $fold --arch $arch --loss CE --alpha 0 &
  done
  wait
done