# CHANGELOG — release3 CRAFT 优化版

## [2026-07-30] dataset2 回退有效配置并重跑（加速）

上一轮 ds2（hidden256 / 64neg / pop_hard）Val@100 卡在 ~0.40。回退：

| 项 | 新值 |
|----|------|
| hidden / heads | **128 / 2** |
| num_neg_train | **30** |
| hard_neg | 历史 **0.5**；**关掉 pop_hard** |
| batch | **192** |
| 其它 | numba CSR、`val_every=2`、向量化 history hard-neg |

只重跑 ds2：`./train_dataset2.sh`（保留已有 dataset1 结果打包）。

---

## [2026-07-30] 训练加速：numba CSR + 大 batch + 隔轮验证

| 改动 | 说明 |
|------|------|
| `fast_ops.py` | CSR + numba：邻居采样 / last-update（约 10–50×） |
| last-update 并入 prefetch | 与 GPU step 重叠 |
| `batch_size` | ds1 **512** / ds2 **96** |
| `val_every=2` | 隔轮 Val@100；早停按验证次数计 |
| Val 只算 MRR | 去掉每 batch 的 sklearn AP/AUC |

```bash
pip install numba   # 若尚未安装
source ./setup_cuda_env.sh && ./train_all.sh
```

---

## [2026-07-30] 修复 dataset1：关闭历史难负样本


高重复图（~72% 下一跳在历史里）把历史当 hard-neg 会压低正确答案 → Val MRR 从 0.68 崩到 ~0.45，而 AP/AUC 仍升。

| 数据集 | 改动 |
|--------|------|
| dataset1 | `hard_neg=0` / `pop_hard=0`，改回 **BPR**，`num_neg_train=10`，`batch=256` |
| dataset2 | 保留历史+流行难负 + hidden=256（低重复 ~2.7%，合理） |
| 共同 | 仍 `num_neg_val=99` 早停 |

---

## [2026-07-30] 训练对齐评测：Val@100 早停 + 流行度难负样本 + ds2 加宽

补齐此前未落地项（需**全量重训**）：

| 项 | 改动 |
|----|------|
| Val 对齐测试 | `num_neg_val=99`，早停用 **MRR@100** |
| 历史难负样本 | `hard_neg_ratio`（ds1=0.4 / ds2=0.3） |
| 流行度难负样本 | `pop_hard_ratio` + `pop_top_k` 按频次采样 |
| dataset1 | `num_neg_train=20`，softmax，batch=128 |
| dataset2 | `hidden=256`，`n_heads=4`，`num_neg_train=64`，batch=48，softmax temp=0.8 |
| 推理集成 | 仍用 best+last checkpoint 加权（未做多 seed，耗时翻倍） |

```bash
source ./setup_cuda_env.sh
./train_all.sh   # 默认 PACK=1 → result.zip
```

---

## [2026-07-30] Val@100 搜索推理超参（不改模型）

榜分 1.14 后：用「1 正 + 99 负」缓存 logits，网格搜索后处理。

| 数据集 | baseline MRR@100 | tuned | 要点 |
|--------|------------------|-------|------|
| dataset1 | 0.836 | **0.883** | 中等 history + 强 cooccur + 轻 pop |
| dataset2 | 0.625 | **0.695** | **关掉** history_boost/pop；保留 count+cooccur；T=0.7 |

脚本：`tune_infer.py` → `saved_models/{ds}_infer_best.json`；`predict.py` 自动加载。

```bash
python -u tune_infer.py --dataset dataset1
python -u tune_infer.py --dataset dataset2
python -u predict.py --dataset dataset1
python -u predict.py --dataset dataset2 --pack
```

---

## [2026-07-30] 推理强化：历史次数/近因 + 共现 + best/last 集成

冲榜（1.13→更高）不重训：加强 logit 融合并集成 checkpoint。

### 推理改动
| 项 | 说明 |
|----|------|
| `history_count_coef` | `log(1+次数)` 加分 |
| `history_recency_coef` | 最近交互时间衰减加分 |
| `cooccur_boost` | 训练边 src–dst 共现 `log(1+c)` |
| `pred_temperature` | Softmax 温度 <1 拉开差距 |
| Ensemble | `predict.py` 默认加权平均 **best + last** logits |

### 默认强度（config.py）
- dataset1：history=3.5 / count=1.2 / recency=2.0 / cooccur=1.5 / pop=0.3 / temp=0.85  
- dataset2：history=2.0 / count=0.8 / recency=1.5 / cooccur=1.2 / pop=1.2 / temp=0.9  

### 用法
```bash
source ./setup_cuda_env.sh
python -u predict.py --dataset dataset1
python -u predict.py --dataset dataset2 --pack
```

---

## [2026-07-30] `train_all.sh` 默认自动打包 result.zip

原先需 `PACK=1` 才打包；现默认 `PACK=1`，训完写出 `./result.zip`。不需要打包时：`PACK=0 ./train_all.sh`。

---

## [2026-07-29] 正确接入 input 时间特征 + 提升 GPU 利用率

### CRAFT `input_cat_time_intervals` 修复
开启后 cross-attn 按 `2*H` 建层，key=邻接 emb∥时间，但 **query（dst emb）仍是 H**，触发 `([B,K,64],[128,128])`。

**修复**（`jittor_geometric/nn/models/craft.py`，site-packages + 本地 JittorGeometric 同步）：
- query 侧同样拼接 dst 时间间隔特征 → `[emb ∥ Δt]`，与 key 同为 `2H`
- `release3` 优先 `sys.path` 加载本地 `JittorGeometric`

### GPU 利用率
| 改动 | 说明 |
|------|------|
| `JT_SYNC` 默认 `0` | 去掉逐步强制同步（原默认 1 把 GPU 拖成串行） |
| 去掉每 step `jt.sync_all()` | 改为 `sync_every`（ds1=20 / ds2=10） |
| 训练 prefetch | GPU 跑当前 batch 时 CPU 预采样下一批邻居 |
| 快速 last-update | `get_dst_last_update_times` 用 `searchsorted` 取最后一次交互，不再走 `num_neighbors=1` 全量采样矩阵 |

### Config
- dataset1 / dataset2：`input_cat_time_intervals=True`（**改结构，需重训**）
- 当前正在跑的 dataset1（无 input time）不受影响；下一次重训才会吃到新配置

### 重训示例
```bash
source ./setup_cuda_env.sh   # JT_SYNC=0
python -u main.py --dataset dataset2 --epochs 100 --early_stop 10
```

---

## [2026-07-29] 修复推理崩溃区间 + 回退 dataset1 + 历史候选加分

### 问题
1. dataset2 提交分落在 **[0.90, 1.00]**：sigmoid 饱和后再做 **概率空间** 流行度混合（`0.9*p + 0.1*pop`）把下限抬死  
2. dataset1 改用 softmax/难负后 Val MRR **0.73**，低于此前 multi-BPR 的 **~0.86**

### 改动
| 项 | 说明 |
|----|------|
| 推理 | 对 100 候选做 **softmax**（可切 `sigmoid`）；`history_boost` / `pop_prior_alpha` 只加在 **logit** 上 |
| dataset1 | 回退：`train_loss=bpr`, `num_neg_train=5`, `hard_neg_ratio=0`, `history_boost=2.0` |
| dataset2 | 保留现有 best（softmax 训出的 hidden=128）；推理改为 softmax + `history_boost=1.0` + logit pop `0.5` |
| `predict.py` | 默认用 `config.py` **刷新推理 knobs**（结构仍读 checkpoint 旁 JSON） |

### 操作
```bash
# 不必重训 dataset2：用现有 best 重推理
source ./setup_cuda_env.sh
python -u predict.py --dataset dataset2 \
  --model_path ./saved_models/dataset2_CRAFT_best.pkl

# dataset1 需按回退配置重训
python -u main.py --dataset dataset1 --epochs 100 --early_stop 10
```

---

## [2026-07-29] 修复 tee 管道下 epoch 日志不刷新

### 原因
`train_all.sh` 用 `python … | tee log` 时，Python 对管道默认**块缓冲**，epoch 的 `print` 要攒满缓冲才写入日志，看起来像“没输出”。

### 修复
- `main.py`：`stdout/stderr` 行缓冲；新增 `log()`（`flush=True`）；每个 epoch 打印 `start` / Train Loss / Val / `done`
- `train_all.sh`：`PYTHONUNBUFFERED=1` + `python -u`

---

## [2026-07-29] 对齐 MRR：sampled softmax + 难负样本 + dataset2 加宽

针对「Val AUC 高、Val MRR 偏低」（尤其 dataset2），把训练目标与多候选排序对齐，并加大 dataset2 容量。

### 代码改动

| 文件 | 说明 |
|------|------|
| `main.py` | `sampled_softmax_loss`；`inject_hard_negatives`（用 src 历史邻居替换部分随机负样本）；`compute_train_loss` 分发 `bpr`/`softmax`；推理阶段 `blend_with_popularity`（log 流行度先验） |
| `config.py` | 新字段 `train_loss` / `hard_neg_ratio` / `pop_prior_alpha`；分数据集超参上调 |
| `predict.py` | 推理同样应用 `pop_prior_alpha` 融合 |
| `CHANGELOG.md` | 本条目 |

### 默认超参（相对上一版）

| 项 | dataset1 旧→新 | dataset2 旧→新 |
|----|----------------|------------------|
| `train_loss` | BPR → **softmax** | BPR → **softmax** |
| `num_neg_train` | 5 → **20** | 5 → **30** |
| `hard_neg_ratio` | 无 → **0.4** | 无 → **0.5** |
| `num_neighbors` | 50（不变） | 30 → **50** |
| `hidden_size` | 64（不变） | 64 → **128** |
| `batch_size` | 200（不变） | 200 → **100**（显存） |
| `pop_prior_alpha` | 0 → **0.05** | 0 → **0.10** |
| CRAFT `loss_type` | 仍为 `BPR` | 仍为 `BPR`（保证 `predict` 返回 **logits**，供 softmax/BPR 训练） |

### CLI 新增覆盖项
```bash
--train_loss {bpr,softmax}
--hard_neg_ratio FLOAT
--pop_prior_alpha FLOAT
--hidden_size INT
```

### 兼容性
- **必须重新训练**：`hidden_size` / `num_neighbors` / 负样本数变化后，旧 `*_CRAFT_best.pkl` 不兼容
- 旧行为可回退：`--train_loss bpr --hard_neg_ratio 0 --pop_prior_alpha 0 --num_neg_train 5`

### 请重新训练
```bash
cd release3
source ./setup_cuda_env.sh
# 或一键：
./train_all.sh
# 单数据集：
python main.py --dataset dataset1 --epochs 100 --early_stop 10
python main.py --dataset dataset2 --epochs 100 --early_stop 10
```

### 仍未做（后续）
- 排除历史已交互边作假负（dataset1 高重复场景需谨慎）
- 多 seed / checkpoint 概率平均
- 正确接入 `input_cat_time_intervals`（需改 CRAFT query 维）
- 换 GraphMixer / TGN

---

## [2026-07-28] 修复 dataset2 `AssertionError: ([200,6,64], [128,128])`

### 原因
`config.py` 里 dataset2 曾设 `input_cat_time_intervals=True`。CRAFT 会把 cross-attention 建为 `2 * hidden_size=128`，但 query（候选 dst embedding）仍是 64 维，于是 `matmul_transpose` 断言失败。

### 修复
- dataset2 改回 `input_cat_time_intervals=False`（与官方 CRAFT 用法一致）
- 时间信息仍通过已开启的 `output_cat_time_intervals=True` 注入

### 请重新跑 dataset2
```bash
source ./setup_cuda_env.sh
python main.py --dataset dataset2 --epochs 100 --early_stop 10
# 或整段重跑：
./train_all.sh
```

---

## [2026-07-27] 修复 CURAND_STATUS_INITIALIZATION_FAILED（无法启动 GPU 训练）

### 原因
1. 部分 AutoDL 节点只挂了 `/dev/nvidia7`，没有 `/dev/nvidia0`
2. 环境里 PyTorch 的 **nvidia-cu13** 自带 `libcurand.so.10`，与 Jittor 使用的 **CUDA 11.8** 冲突，导致 `curandCreateGenerator` 返回 203

### 修复
| 项 | 说明 |
|----|------|
| `train_all.sh` | 启动前自动 `ln -sf` 补齐 `/dev/nvidia0`；`LD_PRELOAD` 强制 CUDA11.8 的 `libcudart`+`libcurand`；训练前做 GPU smoke test |
| `setup_cuda_env.sh` | 手动跑 `main.py` 时可 `source ./setup_cuda_env.sh` |
| Jittor curand 源码 | 改为 **lazy init**（`ensure_curand_ready()`），避免 `.so` 静态构造阶段过早初始化 |

### 手动跑单数据集（务必先 source）
```bash
cd release3
source ./setup_cuda_env.sh
python main.py --dataset dataset1 --epochs 100 --early_stop 10
```

### 一键顺序训练
```bash
./train_all.sh
```

---

## [2026-07-27] 一键顺序训练

### 新增
| 文件 | 说明 |
|------|------|
| `train_all.sh` | 依次训练 dataset1 → dataset2，日志写入 `logs/` |

### 用法
```bash
cd release3
chmod +x train_all.sh
./train_all.sh

# 可选环境变量
EPOCHS=50 EARLY_STOP=5 ./train_all.sh          # 缩短轮数做冒烟
./train_all.sh                                  # 训完默认打包 result.zip
PACK=0 ./train_all.sh                           # 跳过打包
TQDM_DISABLE=0 ./train_all.sh                   # 保留进度条
```

---

## [2026-07-27] 相对官方/原 baseline 的优化

### 动机
原 baseline 用 **1 个随机负样本 + Val AP 早停**，与正式评测（100 候选上的 **MRR**）不对齐；两个数据集共用同一套超参，未利用「非二部高重复 / 二部大图」差异。

### 新增文件
| 文件 | 说明 |
|------|------|
| `config.py` | 分数据集超参表 + `get_dataset_config()` |
| `CHANGELOG.md` | 本改动日志 |

### 修改文件

#### `main.py`
1. **多负样本训练**  
   - `neg_sampling_ratio=1.0` → `num_neg_sample=K`（默认 K=5）  
   - 损失改为 **Multi-Negative BPR**：对每个正样本与 K 个负样本分别算 `-log σ(pos−neg_k)` 再取平均  
2. **负样本去碰撞**  
   - 新增 `avoid_pos_collision()`：若随机负样本与同行正样本 ID 相同则重采样  
3. **验证对齐评测**  
   - 验证采 **49 个负样本**，计算 **MRR**（并用 AP/AUC@1neg 作辅助日志）  
   - **早停指标：Val MRR**（不再用 AP）  
4. **分数据集超参**（见下方表）  
5. **可复现配置落盘**  
   - 每次训练写入 `saved_models/{dataset}_config.json`，供 `predict.py` 重建同结构模型  
6. **CLI 可覆盖**  
   - 新增 `--lr` / `--num_neg_train` / `--num_neg_val` / `--num_neighbors`；`--batch_size` 改为覆盖 config（默认读 config）

#### `predict.py`
1. 从 `{dataset}_config.json`（或 `--config_path`）加载训练时配置  
2. 用 `build_craft_model(cfg, ...)` 建模型，避免与训练结构不一致  
3. 不再写死 `num_neighbors=30, hidden_size=64`

### 默认超参（相对旧 baseline）

| 项 | 旧 baseline | dataset1（新） | dataset2（新） |
|----|-------------|---------------|----------------|
| num_neighbors | 30 | **50** | 30 |
| hidden_size | 64 | 64 | 64 |
| 训练负样本数/正样本 | 1 | **5** | **5** |
| 验证负样本数/正样本 | 1 | **49** | **49** |
| 早停指标 | Val AP | **Val MRR** | **Val MRR** |
| input_cat_time_intervals | False | False | **True** |
| output_cat_repeat_times | True | True | True |
| lr / batch / epochs / patience | 1e-4 / 200 / 100 / 10 | 同左（可由 CLI 改） | 同左 |

### 未改动（刻意保留）
- 模型骨架仍为 **CRAFT**（未换 TGN/GraphMixer）  
- 邻居策略仍为 `recent`  
- 训练/验证时间切分仍为 **后 15% 验证**  
- 提交格式不变：每行 100 个 8 位小数概率  

### 兼容性注意
- **旧 checkpoint（`num_neighbors=30` 等）与新默认 config 可能不兼容**  
  - 用旧模型推理时：需提供训练时对应的 config，或把 `config.py` 临时改回旧值  
  - 推荐：**用新代码重新训练** 再提交  
- 验证 49 负样本会比原来更慢；可用 `--num_neg_val 19` 加速调参  

### 推荐运行命令

```bash
cd release3

# dataset1（更长历史 + 多负样本 + MRR 早停）
python main.py --dataset dataset1 --epochs 100 --early_stop 10

# dataset2（输入时间间隔 + 多负样本）
python main.py --dataset dataset2 --epochs 100 --early_stop 10

# 仅推理（自动读 saved_models/{dataset}_config.json）
python predict.py --dataset dataset1 --model_path ./saved_models/dataset1_CRAFT_best.pkl
```

### 后续可做（尚未实现）
- ~~Hard negative / 排除历史已交互边~~ → **部分完成**：历史邻居难负样本已加；排除假负仍未做  
- ~~频率先验与模型分数融合~~ → **已完成**（logit-space `pop_prior_alpha`）  
- 多 seed / checkpoint 概率平均  
- 换 GraphMixer / TGN 专攻新连接  
- ~~正确接入 `input_cat_time_intervals`~~ → **已完成**（见 [2026-07-29] CRAFT query 维修复）  
- ~~sampled softmax 训练目标~~ → **已完成**
