# 点云降噪赛题 Baseline（可交付版）

## 环境安装
```bash
# 安装计图
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 -y # 确保gcc、g++版本不高于10
conda install -c conda-forge libgomp -y # 确保OpenMP runtime存在

# 安装依赖
python -m pip install -r requirements.txt
pip install jittor numpy trimesh scipy omegaconf point-cloud-utils
```

## 数据准备
1. 将训练数据 `dataset_train.tar.gz` 解压（推荐大盘路径）：
   ```bash
   tar xzf dataset_train.tar.gz -C /root/autodl-tmp
   # 默认配置: /root/autodl-tmp/dataset_train/shapenet/...
   ```

2. 将测试数据 `dataset_test_noisy.zip` 解压到：
   ```bash
   # 默认: /root/workspace/release4/dataset_test_noisy/shapenet/.../noisy.npy
   ```

## 一键训练 + 推理 + 打包（推荐）
```bash
cd /root/workspace/release4/starter_code
source ./setup_cuda_env.sh   # 修复 CURAND / CUDA 路径
./run_all.sh
# 产物：./result.zip（可直接提交）
```

默认开启：
- **v2 训练**：residual + 偏重双向 CD（`cd:2.0, p2s:0.3`）
- **默认推理**：`outer=1 / inner=1 / step_scale=0.8 / blend_alpha=0.85`
  （`out = noisy + α(pred−noisy)`；α↑偏 P2S，α↓偏 CD）

冲分可选：
```bash
SKIP_TRAIN=1 PREDICT_BLEND_ALPHA=0.75 ./run_all.sh   # 更护 CD
SKIP_TRAIN=1 PREDICT_BLEND_ALPHA=0.95 ./run_all.sh   # 更贴面（P2S）
BLEND_ALPHAS="0.75 0.85 0.95" STEP_SCALES="0.8" ./sweep_predict.sh
```

可选：
```bash
SKIP_TRAIN=1 ./run_all.sh                 # 已有 best，只推理+打包
SKIP_PREDICT=1 ./run_all.sh               # 只训练
```

## 训练
```bash
source ./setup_cuda_env.sh
python run.py --task configs/task/train_vm_v2.yaml
```
权重：`experiments/vm_v2/checkpoint_best.pkl`（按验证 CD 保存）。

## 推理（生成提交文件）
```bash
# 单 ckpt
python run.py --task configs/task/predict_vm_v2.yaml

# 多 ckpt 集成（推荐，与 run_all 一致）
export ENSEMBLE_CKPTS="$(python pick_ensemble.py --radius 2)"
SKIP_TRAIN=1 ./run_all.sh
```
结果：`results/dataset_test_noisy/shapenet/.../denoised.npy`

推理超参见 `configs/model/vm_v2.yaml` 的 `predict:` 段。

## 打包提交
```bash
./run_all.sh   # 末尾自动打 result.zip
# 或手动:
# (cd results/dataset_test_noisy && zip -qr ../../result.zip shapenet/)
```

## 提交格式
每个测试样本一个 `denoised.npy`，目录结构与测试集一致，打包为 `result.zip`：
```
result.zip
  shapenet/
    <synset_id>/
      <model_id>/
        denoised.npy    # np.float32, shape (N, 3)
```

## 本地评测（需要 GT 数据，仅组委会持有）
```bash
python evaluate.py \
    --pred_dir ./results/dataset_test_noisy \
    --gt_dir ./test_gt \
    --noisy_dir ./dataset_test_noisy \
    --mesh_dir ./dataset_train \
    --workers 8
```
