#!/usr/bin/env bash
set -e  

python tools/expand_kitti_calib.py \
  --in-calib  rosbag2/calib \
  --out-calib rosbag2/calib_full

python tools/make_pcd_1.py \
  --src-pcd rosbag2/pointclouds \
  --dst-root .

python tools/prepare_kitti_test.py \
  --src rosbag2 \
  --kitti data30/kitti

python tools/create_kitti_test_info.py \
  --root-path data30/kitti \
  --out-dir  data30/kitti \
  --extra-tag kitti_2



python tools/test.py configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py checkpoint/best_epoch.pth \
  --format-only \
  --eval-options "pklfile_prefix=./results/kitti_results" "submission_prefix=./results/kitti_txt"

python visual.py