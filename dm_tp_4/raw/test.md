# L20A DeepSeek-V4-Flash SGLang 测试执行手册

Background
● Model repo: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
● SGLang deepseek v4 flush reference: https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4，测试方式及命令参考这个文档，只测试Basic Configuration中的max_throughput_megamoe_w4a8一种模式即可
● Target local model path: /home/admin/DeepSeek-V4-Flash
● Current image: docker.io/lmsysorg/sglang:latest
● Scale-up goal: after TP4 testing, run TP8 on two nodes and compare TP4 -> TP8 behavior.
● Follow-up goal: after TP4/TP8 testing, run PD disaggregation plus EP with 8 prefill GPUs and 8 decode GPUs across 4 machines.


Scale-Up 测试矩阵与核心原则
核心测试原则：
● 同一组 scale-up 曲线只改变 GPU 规模；模型、dtype / quant、请求分布、并发阶梯、统计口径和框架版本应保持一致。
● 每个 GPU 规模都要同时跑 vLLM 和 SGLang；两者都作为必测框架，不区分优先级。
● 判断 scale-up 不只看总吞吐；必须同时看 tokens/s/GPU、TTFT p99、TPOT p99、queue time、error/timeout。
● 环境、拓扑和版本不单独成章；按每次测试记录到矩阵的 复现信息 和 raw log 中，至少包含 GPU 拓扑、NCCL/CUDA/driver、框架 commit、placement。
● 如果某个规模收益不符合预期，先用 profiling 表定位瓶颈，再写结论。

**节点映射：**
*   **主节点 (Master)**: `10.56.160.38` (所有操作的入口)
*   **工作节点 1**: `10.56.160.40`
*   **工作节点 2**: `10.56.160.36`
*   **工作节点 3**: `10.56.160.34`

**操作原则：**
*   **统一入口**：所有命令均在 `10.56.160.38` 上通过 `ssh` 发起。
*   **无投机解码**：不包含 MTP/EAGLE 参数。
*   **模型路径**：`/home/admin/DeepSeek-V4-Flash`。
*   **容器名称**：`sgl-gpu-final`。

---

## 阶段一：环境准备与模型分发

### 1.1 登录主节点并验证模型
```bash
ssh root@10.56.160.38

# 验证模型是否存在
ls -lh /home/admin/DeepSeek-V4-Flash/config.json
```

### 1.2 模型分发至其他三台机器
```bash
# 从主节点同步模型到其他节点
for ip in 10.56.160.40 10.56.160.36 10.56.160.34; do
    echo "Syncing to $ip ..."
    rsync -avz --progress /home/admin/DeepSeek-V4-Flash root@$ip:/home/admin/
done
```

### 1.3 验证所有节点模型完整性
```bash
for ip in 10.56.160.40 10.56.160.36 10.56.160.34; do
    echo "Checking $ip ..."
    ssh root@$ip "ls -lh /home/admin/DeepSeek-V4-Flash/config.json && du -sh /home/admin/DeepSeek-V4-Flash"
done
```

---

## 阶段二：TP4 基准测试（单节点：10.56.160.38）

### 2.1 进入容器并启动 TP4 服务
```bash
# 在主节点进入容器
ssh root@10.56.160.38
podman exec -it sgl-gpu-final bash

# 设置 MegaMoE 环境变量
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320

# 启动 TP4 服务
python3 -m sglang.launch_server \
  --model-path /data/DeepSeek-V4-Flash \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 4 \
  --quantization w4a8 \
  --dtype bfloat16 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 16384 \
  --max-running-requests 1024 \
  --moe-a2a-backend megamoe \
  --enable-cache-report
```
*(保持此终端运行，新开一个终端窗口执行测试)*

### 2.2 执行 TP4 基准测试
**在本地新开终端，连接主节点执行：**

```bash
ssh root@10.56.160.38
podman exec -it sgl-gpu-final bash

# 测试 4k/500
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --dataset-name random \
  --random-input-len 4096 \
  --random-output-len 500 \
  --num-prompts 1000 \
  --max-concurrency 128 \
  --output-file /data/tp4_4k_500.json

# 测试 32k/500
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --dataset-name random \
  --random-input-len 32768 \
  --random-output-len 500 \
  --num-prompts 1000 \
  --max-concurrency 128 \
  --output-file /data/tp4_32k_500.json

# 测试 128k/500
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --dataset-name random \
  --random-input-len 131072 \
  --random-output-len 500 \
  --num-prompts 500 \
  --max-concurrency 64 \
  --output-file /data/tp4_128k_500.json
```

### 2.3 停止 TP4 服务
*(在服务运行的终端按 `Ctrl+C`)*

---

## 阶段三：TP8 扩展测试（双节点：38 + 40）

### 3.1 启动 TP8 服务

**节点 0 (10.56.160.38):**
```bash
ssh root@10.56.160.38
podman exec -it sgl-gpu-final bash

export MASTER_ADDR=10.56.160.38
export MASTER_PORT=29500
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320

python3 -m sglang.launch_server \
  --model-path /data/DeepSeek-V4-Flash \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 8 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr 10.56.160.38:29500 \
  --quantization w4a8 \
  --dtype bfloat16 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 16384 \
  --max-running-requests 1024 \
  --moe-a2a-backend megamoe \
  --enable-cache-report
```

**节点 1 (10.56.160.40):**
```bash
ssh root@10.56.160.40
podman exec -it sgl-gpu-final bash

export MASTER_ADDR=10.56.160.38
export MASTER_PORT=29500
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320

python3 -m sglang.launch_server \
  --model-path /data/DeepSeek-V4-Flash \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 8 \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr 10.56.160.38:29500 \
  --quantization w4a8 \
  --dtype bfloat16 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 16384 \
  --max-running-requests 1024 \
  --moe-a2a-backend megamoe \
  --enable-cache-report
```

### 3.2 执行 TP8 基准测试
**在主节点 (38) 新开终端执行：**

```bash
ssh root@10.56.160.38
podman exec -it sgl-gpu-final bash

# 测试 4k/500
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --dataset-name random \
  --random-input-len 4096 \
  --random-output-len 500 \
  --num-prompts 1000 \
  --max-concurrency 256 \
  --output-file /data/tp8_4k_500.json

# 测试 32k/500
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --dataset-name random \
  --random-input-len 32768 \
  --random-output-len 500 \
  --num-prompts 1000 \
  --max-concurrency 256 \
  --output-file /data/tp8_32k_500.json

# 测试 128k/500
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --dataset-name random \
  --random-input-len 131072 \
  --random-output-len 500 \
  --num-prompts 500 \
  --max-concurrency 128 \
  --output-file /data/tp8_128k_500.json
```

### 3.3 停止 TP8 服务
*(在两个节点的容器终端分别按 `Ctrl+C`)*

---

## 阶段四：PD 分离 + EP 测试（四节点：38,40 Prefill | 36,34 Decode）

### 4.1 启动 Prefill 服务（节点 38, 40）

**Prefill Node 0 (10.56.160.38):**
```bash
ssh root@10.56.160.38
podman exec -it sgl-gpu-final bash

export MASTER_ADDR=10.56.160.38
export MASTER_PORT=29500

python3 -m sglang.launch_server \
  --model-path /data/DeepSeek-V4-Flash \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 4 \
  --ep-size 4 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr 10.56.160.38:29500 \
  --disaggregation-mode prefill \
  --quantization w4a8 \
  --dtype bfloat16 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 16384 \
  --max-running-requests 3072 \
  --moe-a2a-backend deepep \
  --enable-cache-report
```

**Prefill Node 1 (10.56.160.40):**
```bash
ssh root@10.56.160.40
podman exec -it sgl-gpu-final bash

export MASTER_ADDR=10.56.160.38
export MASTER_PORT=29500

python3 -m sglang.launch_server \
  --model-path /data/DeepSeek-V4-Flash \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 4 \
  --ep-size 4 \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr 10.56.160.38:29500 \
  --disaggregation-mode prefill \
  --quantization w4a8 \
  --dtype bfloat16 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 16384 \
  --max-running-requests 3072 \
  --moe-a2a-backend deepep \
  --enable-cache-report
```

### 4.2 启动 Decode 服务（节点 36, 34）

**Decode Node 0 (10.56.160.36):**
```bash
ssh root@10.56.160.36
podman exec -it sgl-gpu-final bash

export MASTER_ADDR=10.56.160.36
export MASTER_PORT=29501
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=65536

python3 -m sglang.launch_server \
  --model-path /data/DeepSeek-V4-Flash \
  --host 0.0.0.0 \
  --port 30001 \
  --tp-size 4 \
  --ep-size 4 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr 10.56.160.36:29501 \
  --disaggregation-mode decode \
  --quantization w4a8 \
  --dtype bfloat16 \
  --mem-fraction-static 0.78 \
  --cuda-graph-max-bs 256 \
  --max-running-requests 512 \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --enable-cache-report \
  --decode-log-interval 100
```

**Decode Node 1 (10.56.160.34):**
```bash
ssh root@10.56.160.34
podman exec -it sgl-gpu-final bash

export MASTER_ADDR=10.56.160.36
export MASTER_PORT=29501
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=65536

python3 -m sglang.launch_server \
  --model-path /data/DeepSeek-V4-Flash \
  --host 0.0.0.0 \
  --port 30001 \
  --tp-size 4 \
  --ep-size 4 \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr 10.56.160.36:29501 \
  --disaggregation-mode decode \
  --quantization w4a8 \
  --dtype bfloat16 \
  --mem-fraction-static 0.78 \
  --cuda-graph-max-bs 256 \
  --max-running-requests 512 \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --enable-cache-report \
  --decode-log-interval 100
```

### 4.3 执行 PD 分离基准测试
**在主节点 (38) 新开终端执行：**

```bash
ssh root@10.56.160.38
podman exec -it sgl-gpu-final bash

# 测试 PD 分离 4k/500
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --dataset-name random \
  --random-input-len 4096 \
  --random-output-len 500 \
  --num-prompts 1000 \
  --max-concurrency 512 \
  --output-file /data/pd_ep_4k_500.json
```

---

## 阶段五：结果收集

### 5.1 从各节点收集测试结果
```bash
# 在主节点创建结果目录
mkdir -p ~/test_results

# 收集主节点结果
cp /home/admin/*.json ~/test_results/ 2>/dev/null || true

# 从其他节点拉取结果
for ip in 10.56.160.40 10.56.160.36 10.56.160.34; do
    scp root@$ip:/home/admin/*.json ~/test_results/ 2>/dev/null || true
done

# 查看收集到的文件
ls -lh ~/test_results/
```

---

## 常见问题排查

1.  **NCCL 通信失败**：
    *   检查防火墙是否放行 `29500` 和 `29501` 端口。
    *   设置 `export NCCL_DEBUG=INFO` 查看详细日志。
2.  **DeepEP 报错**：
    *   检查 `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` 是否满足 `max-running-requests * topk`。
3.  **OOM**：
    *   降低 `--mem-fraction-static` 或 `--max-running-requests`。

---

## 执行记录：TP4 单机 NVLS 复测

时间：2026-05-28

结论：单机 TP4 `32k_500` case 中，真实 NVLS 模式没有看到稳定性能提升；后续单机 NVLS 不再继续扩展测试。

说明：
* 仅配置 `NCCL_P2P_LEVEL=NVL NCCL_NVLS_ENABLE=1 NCCL_PROTO=LL128` 时，服务日志中 `enable_nccl_nvls=False`，未实际开启 SGLang NVLS。
* 真实 NVLS 复测使用 `--enable-nccl-nvls`，服务日志确认 `enable_nccl_nvls=True`。
* `c128` 在用户要求停止单机 NVLS 后终止，未纳入结论。

32k/500 对比结果（非 NVLS vs 真实 NVLS）：

| concurrency | off req/s | on req/s | req delta | off out tok/s | on out tok/s | out delta | off p99 TTFT ms | on p99 TTFT ms | TTFT delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.357 | 0.350 | -2.1% | 89.0 | 87.2 | -2.1% | 1513 | 1425 | -5.8% |
| 4 | 0.871 | 0.840 | -3.5% | 217.3 | 209.6 | -3.5% | 1488 | 1465 | -1.5% |
| 8 | 1.257 | 1.210 | -3.7% | 313.6 | 301.9 | -3.7% | 1434 | 1532 | +6.8% |
| 16 | 1.712 | 1.743 | +1.8% | 427.1 | 434.7 | +1.8% | 2804 | 3439 | +22.6% |
| 32 | 2.325 | 2.245 | -3.4% | 579.9 | 560.0 | -3.4% | 5213 | 6394 | +22.7% |
| 64 | 3.514 | 3.248 | -7.6% | 876.5 | 810.2 | -7.6% | 7358 | 10248 | +39.3% |
