alphas=(0 0.2 0.4 0.6 0.8 1.0 2.0)
architectures=(transformer wbcmil)

for alpha in "${alphas[@]}"; do
  for arch in "${architectures[@]}"; do
    python analyze_script.py --alpha "$alpha" --arch "$arch"
  done
done