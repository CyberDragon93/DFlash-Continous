# DFlash-Continuous: Can a Continuous Flow Drafter Beat Discrete Masked Drafting?

对 [DFlash](https://arxiv.org/abs/2602.06036)(块扩散投机解码)的一个系统性研究:把它的**离散 masked drafter 换成真正的连续流(flow matching / ODE)**,在完全公平的对照下测量接受长度 τ。目标模型 Qwen3-4B,单节点 4×B200。

## TL;DR

1. **连续流 drafter 能达到、但从未超过离散 mask 基线**(τ 6.99 vs 7.05),确定性多步 ODE 精炼在 temp-0 贪心验证下**零增益**——被 K 扫描、harden 对照、flip 统计三重锁死。根因:KV 注入的 target 隐层特征已榨干条件信息(首位命中率全臂饱和于 ~89%),块内再"多看几眼"没有新信息。这定量印证了 DFlash 单步设计的合理性。
2. 真正可迁移的两个收益,与"连续 vs 离散"无关:
   - **目标对齐数据**:100K temp-0 贪心重生成数据续训,GSM8K τ 6.15 → 7.05;
   - **Best-of-N 后验采样验证**:候选 0=argmax 保底 + N-1 个温度采样候选,单次批量 target 前向验证取最优。τ 7.05 → **8.72**(N=32),tok/s 不降反升;对官方 checkpoint 零改动即 +1.0。
3. 纯 ELF 配方是唯一多步机制真实有效的臂(K=1→2:+0.58),但绝对天花板低(≤5.0);热启动后多步反而有害。

## Round 2:fork/commit 突破(τ 7.05 → 11.0,零额外验证成本)

第二轮自动研究的问题:什么机制能真正超越单 draft τ≈7.0 的条件信息天花板?答案是 **commit 训练 + branch-recomplete 解码** 与 **block-32** 的叠加:

| 配置(GSM8K-200, max_new 512) | τ | tok/s |
|---|---|---|
| mask 块16 确定性(基线) | 7.05 | 58.5 |
| + 边际采样 best-of-16 | 8.38 | — |
| mask_commit + branch-recomplete N=16 | 9.17 | — |
| mask32_commit + brc N=32 | **10.19** | 74.8 |
| mask32_commit + brc N=64 | **11.03** | 68.8 |
| 跨域(N=32):Math500 / HumanEval / MT-Bench | **11.24 / 8.77 / 4.29** | — |

机制三部曲(每步有对照收据):
1. **P3 回放探针**:未训练 drafter 被块内提交前缀毒害(c=8 时首自由位准确率 −0.067)——块内非 mask token 是分布外输入
2. **commit 训练**(50% 块随机提交真前缀,CE 只算后缀):U(c) 翻转为 c=8 时 **+0.330**(0.55→0.88)
3. **branch-recomplete 解码**:候选 i = argmax 前缀 + 第 i 个最低 margin 位置翻到第 r 优 token + 提交后**重补全尾部**(共 2 次 draft 前向);批量验证取最优

### 连续流的终审(本轮补齐的机制性证据)

- **flow-joint 联合采样死刑**:贪心数据采样器 0% 候选去重;temp-0.8 数据 ELF 采样器 6.7% 去重(15 个噪声种子解码出同一块)。根源链:CE 训练→噪声不变性;MSE 训练→条件均值坍缩;共同根源:KV 注入条件几乎决定答案(decode CE 0.007),潜噪声通道冗余被剪。**强条件 drafter 的潜噪声多样性在原理上不可行**;可用多样性只存在于解码后验(边际采样)与训练过的前缀提交分支(brc)
- flow 多步 @ 块32:K2−K1 = +0.04(第 4 次证伪)
- **冠军机制上的 flow 孪生 rcf_commit:brc N=32 = 9.75 vs mask 9.87** —— 第 5 次机制等价,flow 全程未能超越 mask
- 零训练解码点子全灭:退火(双向)、强制偏离 strat、T 扫描、无重补全的翻转,全部 ≤ 边际采样 8.38

## Round 1 大规模基准(τ,贪心验证,max_new 1024,协议与 dflash 官方 benchmark 模板一致)

| 数据集 (n) | z-lab 官方 | mask_ws | **rcf_v2 flow** | mask+bof16 | **rcf+bof16** |
|---|---|---|---|---|---|
| GSM8K (1000) | 6.25 | 7.14 | 7.09 | **8.49** | 8.47 |
| Math500 (400) | 7.60 | 8.05 | 8.05 | 9.47 | **9.48** |
| HumanEval (164) | 6.43 | 6.37 | 6.36 | 7.81 | **7.83** |
| MBPP (257) | 5.71 | 5.87 | 5.86 | **7.27** | 7.23 |
| MT-Bench (80) | 3.22 | 3.30 | 3.30 | **4.16** | 4.16 |

三个模式:①连续流与离散 mask 在所有 5 个域逐位打平;② best-of-16 在**每个域** +1.3~1.5 τ(+19%~26%),且端到端 tok/s 持平或更快;③我们的 100K 数据在数学域大胜官方(+0.9),代码/对话域持平。

## 详细结果(第一轮,GSM8K-200 / Math500-100)

### GSM8K-200 / Math500-100(第一轮)

| 方案 | GSM8K τ | Math500 τ |
|---|---|---|
| z-lab 官方 ckpt(800K 数据) | 6.15 | 7.53 |
| mask 从零(我们 100K) | 5.63 | 5.80 |
| mask 续训(我们 100K) | 7.05 | 7.97 |
| RCF 连续流 K=1 | 6.99 | 7.93 |
| RCF K=2/3/4/8、harden、T_r 0.5 | 全部 ≈7.0 | ≈7.94 |
| ELF 连续流(从零)K=1/2/4 | 3.51 / 4.08 / 4.05 | 3.48 / 3.98 / 3.88 |
| ELF(热启动)K=1/2/4 | 5.04 / 4.40 / 4.44 | — |

### Best-of-N 后验采样(T=0.8,候选 0 = argmax)

| drafter | N=1 | N=4 | N=8 | N=16 | N=32 |
|---|---|---|---|---|---|
| mask(离散) | 7.05 | — | 7.98 | 8.36 | 8.72 |
| RCF flow(连续) | 6.99 | 7.55 | 7.97 | 8.36 | 8.69 |

逐格相等 → 收益是解码策略,不是流。输入噪声作分支选择器(flow 独有路线)实测失败(N=8 仅 +0.06):CE 训练教出噪声不变性,且 temp-0 数据是点分布。

## 三种模式(一套框架,公平对照)

- `--mode mask` — DFlash 离散基线复现(训练侧:随机锚点 + 位置衰减 CE)
- `--mode flow` — [ELF](https://arxiv.org/abs/2605.10938) 配方:x-prediction 流匹配、80/20 去噪(纯速度 MSE)/解码(t=1 扰动输入 CE)双分支、self-conditioning、logit-normal t、SDE 采样
- `--mode rcf` — Rollout-Consistent Flow(29-agent 头脑风暴收敛设计):endpoint = 冻结 lm_head 的 top-64 后验混合(零新增输出参数)、DDIM 型积分器、**rollout 训练**(t>0 状态由模型自身 pass-1 预测构造,消除 oracle 插值失配)、TE 偏置=mask embedding + z-lab 热启动 → 初始时 K=1 逐位等于官方 DFlash(U1 检验)

## 使用

```bash
# 1. 数据:用目标模型贪心重生成 100K 开放 prompt(vLLM,GB200 注意 FLASH_ATTN + 禁 flashinfer 采样器)
python cflow/data/prep_prompts.py --out data/prompts.jsonl
sbatch scripts/datagen.sbatch

# 2. 训练(单节点 4 卡 DDP,~1-2h)
torchrun --nproc_per_node=4 -m cflow.train --data data/train_raw.jsonl \
    --mode rcf --name rcf_v1 --warm-start zlab --epochs 4 --lr 2e-4 \
    --weight-decay 0.01 --beta2 0.95

# 3. 评测 τ(支持 z-lab 官方 ckpt 直接加载做参照)
python -m cflow.eval_tau --draft runs/rcf_v1/ckpt.pt --dataset gsm8k --n 200 \
    --steps 2                                  # K 步 ODE
python -m cflow.eval_tau --draft runs/mask_ws/ckpt.pt --dataset gsm8k --n 200 \
    --candidates 16 --cand-mode sample --cand-temp 0.8   # best-of-N
```

## 复现要点 / 踩坑记录

- **变长 batch 是 20 倍吞吐杀手**:每个 batch 形状不同 → cuBLAS/分配器每步冷启动(3s/micro)。长度分桶 + 固定 pad 桶后 0.16s/micro。
- HF 目标模型前向**不要传 attention_mask**(右 padding 因果安全),否则掉出 flash 路径。
- RCF 的随机分支(p_roll/depth2)必须用跨 rank 一致的 RNG,否则 DDP allreduce 死锁。
- GB200 计算节点无 nvcc:vLLM 需 `attention_backend="FLASH_ATTN"` + `VLLM_USE_FLASHINFER_SAMPLER=0`。
- 评测停止符必须包含 bonus token 位置(与 dflash 逐位对齐);`.to(bf16)` 会连 rotary inv_freq 一起转,需只转参数。

`results/` 内含全部评测 JSON(τ、逐 offset 接受率、flip 统计、best-of-N 胜者统计)。

## 致谢

基于 z-lab 的 [DFlash](https://github.com/z-lab/dflash)(推理语义逐位对拍)与 MIT 的 [ELF](https://arxiv.org/abs/2605.10938)(连续流配方)。
