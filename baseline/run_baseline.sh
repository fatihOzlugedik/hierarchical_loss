#!/bin/bash
for arch in transformer wbcmil; do
  for fold in {0..4}; do
    python run_pipeline.py --fold $fold --arch $arch &
  done
  wait
done