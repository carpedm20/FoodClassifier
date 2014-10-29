#!/usr/bin/env sh
# Compute the mean image from the food100 training leveldb
# N.B. this is available in data/food100

./build/tools/compute_image_mean examples/food100/food100_train_lmdb \
  data/food100/food100_mean.binaryproto

echo "Done."
