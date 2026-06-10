# EdgeAI-ROS / EdgeAI-OS Runtime：後續研究深化建議報告

## 1. 報告目的

本報告整理目前 EdgeAI-ROS / EdgeAI-OS Runtime 專題在完成基本 ViT、LLM、concurrent workload、OS resource monitoring、adaptive scheduling、stress testing 與 Docker edge simulation 之後，若要進一步提升研究深度，可以延伸的實作方向。

目前專題已經具備：

- ViT / RepViT runtime scheduling
- LLM context / token budget control
- CPU / memory / page fault / context switch / I/O monitoring
- rule-adaptive 與 predictive-adaptive scheduling
- admission control
- CPU / memory / I/O stress testing
- concurrent ViT + LLM workload
- Docker cgroups edge simulation
- ROS 2 RepViT pipeline 實驗基礎

下一階段若希望期末 project 更接近研究型系統專題，重點不應只是增加更多 benchmark，而應該從：

> 觀察 latency 是多少

進一步升級為：

> 解釋 latency 從哪裡來，並提出 OS-level mechanism 去改善它

因此，本報告建議的方向會聚焦於：

1. deadline miss decomposition
2. true model + device switching
3. CPU affinity / priority / OS scheduling control
4. queueing theory 與 admission control
5. ROS 2 QoS / DDS middleware 實驗
6. tracing-based bottleneck analysis
7. accuracy-latency Pareto frontier
8. hybrid CPU-GPU offloading scheduler
9. model cache / cold start / lazy loading

---

## 2. 現有成果定位

目前專題可以分成兩條主線。

### 2.1 EdgeAI-OS Runtime

此方向偏向通用 OS-aware runtime。

目前已支援：

- OS resource monitor
- EWMA latency predictor
- predictive utility-based scheduler
- admission controller
- ViT workload
- LLM workload
- concurrent ViT + LLM workload
- stress injection
- Docker cgroups simulation

此方向的核心價值是：

> 將 AI inference 視為 OS-level resource scheduling 問題，而不是單純模型 benchmark。

### 2.2 EdgeAI-ROS RepViT

此方向偏向 ROS 2 edge perception pipeline。

目前已支援：

- image publisher
- adaptive RepViT node
- edge AI logger
- CPU / GPU 比較
- deadline experiment
- policy comparison
- stress experiment
- RepViT model family benchmark
- real-image workload
- official RepViT checkpoint 評估

此方向的核心價值是：

> 在實際 ROS 2 pipeline 中評估 RepViT 的 deadline reliability、latency jitter 與 OS pressure 影響。

---

## 3. 為什麼還需要更深入？

目前已有結果可以證明：

- static_large latency 高、miss rate 高
- static_small latency 低，但品質 proxy 低
- rule_adaptive 可以取得較好的 quality-latency trade-off
- predictive scheduler 需要 careful utility design
- memory stress 會增加 page faults 與 P99 latency
- CPU stress 會導致 admission control defer 大量 request
- concurrent workload 會暴露 ViT + LLM resource contention
- GPU 能顯著降低 RepViT family 的 latency 與 deadline miss

但目前仍有幾個未完全回答的研究問題：

1. deadline miss 的主要來源是模型 inference，還是 ROS / OS pipeline overhead？
2. adaptive policy 只調解析度是否足夠？是否應該動態切換不同 RepViT model？
3. CPU affinity、process priority、real-time scheduling 是否能降低 tail latency？
4. ROS 2 QoS depth 與 reliability policy 是否會影響 freshness 與 latency？
5. GPU 加速後，系統瓶頸是否從模型運算轉移到 middleware / preprocessing / logging？
6. true model switching 是否會帶來 cold-start 與 memory-cache trade-off？
7. LLM 與 ViT 同時執行時，是否需要 priority-aware runtime？

這些問題若能回答，專題就會從「完整實作」升級成「具研究深度的系統分析」。

---

# 4. 研究深化方向一：Deadline Miss Decomposition

## 4.1 動機

目前實驗中主要觀察：

- `infer_ms`
- `e2e_ms`
- `deadline_miss`
- CPU usage
- memory usage
- page faults
- context switches

但若某一張 frame deadline miss，目前還無法精準回答：

- 是 image loading 太慢？
- 是 ROS message passing 太慢？
- 是 preprocessing / resize 太慢？
- 是 RepViT inference 太慢？
- 是 logging I/O 太慢？
- 是 OS scheduling jitter？
- 是 Docker/container overhead？

因此，下一步應該將 end-to-end latency 分解成多個階段。

## 4.2 建議實作

在 ROS 2 pipeline 中加入多個 timestamp。

### image_publisher

```text
t0: image read start
t1: image read end
t2: message publish time
```

### adaptive_repvit_node

```text
t3: message received
t4: preprocess start
t5: preprocess end
t6: inference start
t7: inference end
t8: result publish
```

### edgeai_logger

```text
t9: result received
t10: csv write complete
```

## 4.3 可計算指標

```text
image_load_ms        = t1 - t0
publish_overhead_ms  = t3 - t2
preprocess_ms        = t5 - t4
inference_ms         = t7 - t6
postprocess_ms       = t8 - t7
logger_receive_ms    = t9 - t8
logging_ms           = t10 - t9
e2e_ms               = t10 - t0
non_model_overhead   = e2e_ms - inference_ms
```

進一步可以計算：

```text
inference_ratio      = inference_ms / e2e_ms
preprocess_ratio     = preprocess_ms / e2e_ms
communication_ratio  = publish_overhead_ms / e2e_ms
logging_ratio        = logging_ms / e2e_ms
```

## 4.4 實驗設計

| 實驗 | 目的 |
|---|---|
| CPU vs GPU decomposition | 觀察 GPU 加速後瓶頸是否轉移 |
| stress vs no stress | 觀察 CPU / memory / I/O stress 主要影響哪一段 |
| static_large vs adaptive | 觀察 adaptive 是否只降低 inference latency |
| logger on/off | 觀察 CSV I/O 對 tail latency 的影響 |
| image size 160/192/224 | 觀察 resize/preprocess cost 是否明顯 |

## 4.5 預期研究貢獻

可能得到的結論：

> GPU 將 inference latency 降低後，non-model overhead 佔比上升，代表 edge AI pipeline 的瓶頸從模型運算轉移到 ROS message passing、preprocessing 或 logging。

這類結論比單純報告 latency 數字更有研究深度。

---

# 5. 研究深化方向二：True Model Switching

## 5.1 動機

目前 adaptive RepViT policy 主要針對單一模型，例如 RepViT-M0.9，改變輸入解析度：

| Level | Image size |
|---:|---:|
| 0 | 160 |
| 1 | 192 |
| 2 | 224 |

但實際 edge AI runtime 不一定只調解析度，也可能切換不同模型大小：

```text
M0.6
M0.9
M1.0
M1.1
M1.5
M2.3
```

你目前 RepViT model family 結果已經顯示：

- CPU 上 M0.6 / M0.9 / M1.0 / M1.1 較適合 200 ms deadline
- M1.5 開始明顯 miss
- M2.3 在 CPU 上 miss rate 很高
- GPU 可大幅降低大模型 latency
- M2.3 在 GPU 上仍能接近滿足 200 ms deadline

因此，很適合進一步做 true model switching。

## 5.2 新的 runtime config 設計

可將每個 config 定義為：

```text
(model_name, image_size, device, quality_score)
```

例如：

| Level | Model | Image size | Device | Quality proxy |
|---:|---|---:|---|---:|
| 0 | RepViT-M0.6 | 160 | CPU | 0.65 |
| 1 | RepViT-M0.9 | 192 | CPU | 0.78 |
| 2 | RepViT-M1.1 | 224 | CPU/GPU | 0.84 |
| 3 | RepViT-M1.5 | 224 | GPU | 0.92 |
| 4 | RepViT-M2.3 | 224 | GPU | 1.00 |

## 5.3 Scheduler 策略

```text
如果 CPU 且 deadline 嚴格：
    選 M0.6 / M0.9

如果 GPU 可用且系統壓力低：
    選 M1.5 / M2.3

如果 CPU stress 高：
    切 M0.6 + 160 或 drop/defer

如果 GPU memory 壓力高：
    切回 M0.9 / M1.1

如果 predicted latency 超過 deadline：
    降模型或降解析度
```

## 5.4 建議新增 ModelPool

```python
class ModelPool:
    def __init__(self):
        self.models = {}
        self.active_model = None

    def load_model(self, name, device):
        ...

    def get_model(self, name, device):
        key = (name, device)
        if key not in self.models:
            self.models[key] = self.load_model(name, device)
        return self.models[key]
```

## 5.5 新增指標

```text
model_switch_count
model_load_latency_ms
model_cache_hit
model_cache_miss
memory_rss_after_switch
gpu_memory_after_switch
deadline_miss_after_switch
```

## 5.6 研究問題

1. true model switching 是否比 resolution-only adaptation 更有效？
2. 預載全部模型與 lazy loading 的 latency-memory trade-off 是什麼？
3. model cache size 對 deadline miss 有什麼影響？
4. GPU 上是否值得使用較大模型？
5. 在 CPU stress 下，model switching 是否仍足以維持 deadline？

---

# 6. 研究深化方向三：CPU Affinity / Priority / OS Scheduling Control

## 6.1 動機

目前結果顯示，當 CPU stress 很高時，即使模型降級也可能無法滿足 deadline。這代表：

> application-level adaptation 不一定足夠，需要 OS-level scheduling control。

因此，可以加入：

- CPU affinity
- process priority
- Linux nice value
- real-time scheduling
- Docker CPU quota / cpuset
- ROS node executor/thread control

## 6.2 實作模式

### Mode A：Default Linux Scheduling

```bash
python adaptive_repvit_node.py
```

### Mode B：CPU Affinity

將 inference node 綁定到特定 CPU core：

```bash
taskset -c 2,3 python adaptive_repvit_node.py
```

Python 內也可設定：

```python
import os
os.sched_setaffinity(0, {2, 3})
```

### Mode C：Process Priority / nice

```bash
nice -n -10 python adaptive_repvit_node.py
```

Python 內也可設定：

```python
import os
os.nice(-5)
```

### Mode D：Real-time Scheduling

若權限允許：

```bash
sudo chrt -f 20 python adaptive_repvit_node.py
```

Docker 內可能需要：

```bash
--cap-add=sys_nice
--ulimit rtprio=99
```

## 6.3 實驗矩陣

| Setting | CPU stress | Affinity | Nice | RT scheduling |
|---|---|---|---|---|
| baseline | no | no | no | no |
| stress default | yes | no | no | no |
| stress affinity | yes | yes | no | no |
| stress nice | yes | no | yes | no |
| stress RT | yes | yes | yes | yes |

## 6.4 評估指標

```text
avg latency
P95 latency
P99 latency
max latency
deadline miss rate
voluntary context switches
involuntary context switches
CPU migrations
CPU utilization per core
```

## 6.5 研究問題

1. CPU affinity 是否能降低 context switching？
2. nice priority 是否能降低 tail latency？
3. real-time scheduling 是否能減少 deadline miss？
4. application-level adaptive inference 與 OS-level scheduling control 是否有互補效果？

---

# 7. 研究深化方向四：Queueing Theory + Admission Control

## 7.1 動機

目前實驗多以單張 frame latency 為主，但 real-time perception pipeline 更像 queueing system。

可定義：

```text
arrival rate λ：image publisher 發布速度
service rate μ：RepViT node 處理速度
utilization ρ = λ / μ
```

若：

```text
ρ >= 1
```

queue 會累積，end-to-end latency 會快速上升。

## 7.2 可比較的 Queue Policy

| Policy | 說明 |
|---|---|
| FIFO | 每張 frame 都處理，但可能累積 stale frame |
| latest-only | 只處理最新 frame，丟掉舊 frame |
| deadline-aware drop | 預測會 miss deadline 就 drop |
| adaptive degrade | 降模型或降解析度 |
| hybrid | degrade + drop + defer |

## 7.3 實驗設計

| Publisher rate | QoS depth | Queue policy | Policy |
|---:|---:|---|---|
| 5 Hz | 1 | latest-only | adaptive |
| 10 Hz | 1 | latest-only | adaptive |
| 15 Hz | 5 | FIFO | adaptive |
| 30 Hz | 10 | FIFO | static_large |
| 30 Hz | 1 | latest-only | adaptive |
| 30 Hz | 1 | deadline-aware drop | adaptive |

## 7.4 新增指標

```text
queue_length
dropped_frames
deferred_frames
stale_frame_age_ms
freshness_ms
e2e_latency_ms
deadline_miss_rate
effective_fps
```

其中：

```text
freshness_ms = inference_end_time - image_capture_time
```

## 7.5 研究貢獻

可以回答：

> 對 real-time perception 而言，處理每一張 frame 不一定最好。保持 frame freshness 可能比完整處理所有 frame 更重要。

這個方向非常適合 OS / real-time system 專題。

---

# 8. 研究深化方向五：ROS 2 QoS / DDS Middleware 實驗

## 8.1 動機

ROS 2 的 QoS 設定本身就是 middleware-level scheduling / buffering policy。

可以比較：

```text
BEST_EFFORT vs RELIABLE
KEEP_LAST depth=1 / 5 / 10
VOLATILE durability
```

這會讓 project 更貼近 ROS 2 與 OS middleware。

## 8.2 實作範例

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT
)
```

## 8.3 實驗設定

| QoS | Depth | Reliability | 預期 |
|---|---:|---|---|
| low latency | 1 | BEST_EFFORT | freshness 高，drop 多 |
| balanced | 5 | BEST_EFFORT | 折衷 |
| reliable | 10 | RELIABLE | drop 少，但 stale frame 可能多 |
| strict latest | 1 | RELIABLE | 保證傳送但可能 backpressure |

## 8.4 研究問題

1. QoS depth 越大，是否降低 frame drop，但增加 e2e latency？
2. BEST_EFFORT 是否更適合 real-time camera？
3. RELIABLE 是否在壓力下造成 stale frames？
4. QoS policy 與 adaptive inference policy 是否互相影響？

---

# 9. 研究深化方向六：Tracing-Based Bottleneck Analysis

## 9.1 動機

目前 CSV logging 是 application-level measurement。若要更接近系統研究，可以加入 OS / ROS tracing。

可使用：

```text
ros2_tracing
LTTng
perf
psutil
```

## 9.2 可 trace 事件

```text
callback_start
callback_end
publish
subscribe
executor_spin
thread scheduling
context switch
page fault
CPU migration
```

## 9.3 輕量版：perf stat

若不想整合 ros2_tracing，可先使用：

```bash
perf stat -e context-switches,cpu-migrations,page-faults,cache-misses   python adaptive_repvit_node.py
```

若 Docker 裡無法使用 perf，可在 host 上執行或使用 psutil 指標替代。

## 9.4 研究問題

1. deadline miss frames 是否伴隨更多 context switches？
2. CPU stress 下，callback waiting time 是否增加？
3. GPU 模式是否降低 inference，但增加 CPU-GPU synchronization overhead？
4. logging I/O 是否造成 tail latency spike？

---

# 10. 研究深化方向七：Accuracy-Latency Pareto Frontier

## 10.1 動機

目前真實圖片只有少量樣本，主要作為 sanity check，不足以正式評估 accuracy。

若要更有研究深度，可以建立小型 labeled dataset：

```text
100 images
500 images
1000 images
```

不一定要完整 ImageNet validation，但至少要能估計不同 config 的 relative accuracy。

## 10.2 實驗設定

測試：

```text
M0.6 / M0.9 / M1.1 / M1.5 / M2.3
160 / 192 / 224
CPU / GPU
```

## 10.3 指標

```text
top-1 accuracy
top-5 accuracy
avg latency
P95 latency
P99 latency
deadline miss rate
params
MACs
```

## 10.4 Pareto Frontier

畫圖：

```text
x-axis: latency / P95 latency
y-axis: accuracy
point size: params
color: device
```

目標：

> 在 deadline constraint 下，選擇 Pareto-optimal configuration。

這會讓 `quality_score` 不再只是 proxy，而是可以根據實際 accuracy 校正。

---

# 11. 研究深化方向八：Hybrid CPU-GPU Offloading Scheduler

## 11.1 動機

目前 CPU/GPU 結果已經顯示：

- GPU 對大模型加速更明顯
- M2.3 在 CPU 上 miss rate 很高
- M2.3 在 GPU 上可以接近滿足 200 ms deadline

因此可以做 device-aware scheduler。

## 11.2 Config 設計

```text
(model, image_size, device)
```

例如：

| Config | Model | Size | Device |
|---|---|---:|---|
| C0 | M0.9 | 160 | CPU |
| C1 | M0.9 | 224 | CPU |
| C2 | M1.1 | 224 | GPU |
| C3 | M1.5 | 224 | GPU |
| C4 | M2.3 | 224 | GPU |

## 11.3 實作方式

模型池分別管理 CPU / GPU models：

```python
model_cpu = load_model(model_name, device="cpu")
model_gpu = load_model(model_name, device="cuda")
```

推論時根據 config：

```python
x = x.to(config.device)
model = model_pool.get(config.model_name, config.device)
```

## 11.4 研究問題

1. GPU 是否總是比較好？
2. GPU warm-up 與 transfer overhead 是否會影響短任務？
3. CPU stress 下，GPU offloading 是否能隔離 CPU contention？
4. GPU memory 壓力下，是否應該退回小模型或 CPU？
5. 大模型是否應該只在 GPU 上使用？

---

# 12. 研究深化方向九：Model Cache / Cold Start / Lazy Loading

## 12.1 動機

如果加入 true model switching，就一定會遇到模型載入與快取問題。

可比較：

| Strategy | 說明 |
|---|---|
| preload_all | 啟動時載入全部模型 |
| lazy_load | 第一次用到才載入 |
| lru_cache | 只保留最近 N 個模型 |
| memory_aware_cache | memory 壓力高時釋放大模型 |

## 12.2 新增指標

```text
cold_start_ms
model_switch_ms
model_cache_hit
model_cache_miss
cache_hit_rate
RSS memory
GPU memory
deadline_miss_after_switch
```

## 12.3 研究問題

1. 預載模型是否能降低 first-frame latency？
2. lazy loading 是否造成 deadline miss spike？
3. cache size 對 memory footprint 與 latency 有何影響？
4. memory-aware eviction 是否能降低 page faults？
5. GPU memory 不足時如何選擇模型？

---

# 13. 最推薦的實作組合

如果希望專題很有研究深度，但仍可控，建議選以下三個作為主要升級項目。

## 13.1 必做 A：Deadline Miss Decomposition

理由：

- 最直接提升分析深度
- 能回答 deadline miss 從哪裡來
- 不需要大幅改模型
- 很 OS / ROS system

## 13.2 必做 B：True Model + Device Switching

理由：

- 利用你已經完成的 RepViT family CPU/GPU 結果
- 從 resolution-only adaptation 升級為 model/device-aware scheduler
- 研究價值高

## 13.3 必做 C：CPU Affinity / Priority / Queue Policy

理由：

- 最有 OS 味道
- 可以展示 application-level adaptation 與 OS-level control 的互補
- 能改善 CPU stress 下的 tail latency

## 13.4 選做 D：Accuracy-Latency Pareto Frontier

理由：

- 能把 quality_score 從 proxy 變成實際 accuracy
- 報告會更像 research paper
- 需要準備 labeled images，但不一定要很大

---

# 14. 建議最終題目

## 中文題目

**EdgeAI-ROS Runtime：結合延遲分解、異質裝置調度與 OS 排程控制的 RepViT 即時推論系統**

## English Title

**EdgeAI-ROS Runtime: Deadline Decomposition, Heterogeneous Scheduling, and OS-Level Control for Real-Time RepViT Inference**

---

# 15. 建議最終系統架構

```text
edgeai_ros_runtime/
├── nodes/
│   ├── image_publisher.py
│   ├── adaptive_repvit_node.py
│   ├── runtime_scheduler.py
│   └── edgeai_logger.py
├── runtime/
│   ├── monitor.py
│   ├── predictor.py
│   ├── model_pool.py
│   ├── device_scheduler.py
│   ├── queue_policy.py
│   └── latency_decomposer.py
├── launch/
│   ├── baseline.launch.py
│   ├── adaptive.launch.py
│   └── stress.launch.py
├── scripts/
│   ├── run_deadline_decomposition.sh
│   ├── run_model_switching.sh
│   ├── run_os_scheduling.sh
│   └── summarize_results.py
└── reports/
```

---

# 16. 最終研究主張

完成上述深化後，報告可以主張：

1. **Model inference is not the only bottleneck.**  
   GPU 加速後，preprocess、ROS communication、logging 與 scheduling overhead 佔比上升。

2. **Adaptive inference must be combined with OS-level control.**  
   只降解析度或切小模型，在 mixed stress 下仍可能 miss；需要 CPU affinity、priority、drop/defer 與 queue policy。

3. **Device-aware scheduling is necessary.**  
   大模型在 CPU 上 miss rate 高，但在 GPU 上可滿足 deadline，因此 scheduler 應同時考慮 model size 與 device placement。

4. **Queue freshness matters more than processing every frame.**  
   對 real-time perception 來說，latest-frame policy 可能比 FIFO 更適合。

5. **Predictive scheduling must be deadline-feasible, not only quality-aware.**  
   先前 v2/v3 實驗已顯示 utility design 會直接影響 scheduler 是否能滿足 deadline。

---

# 17. 建議開發順序

## Phase 1：Latency decomposition

```text
1. 加入 t0 ~ t10 timestamps
2. 修改 logger CSV 欄位
3. 寫 decomposition summary script
4. 跑 CPU/GPU/no-stress/stress
```

## Phase 2：True model switching

```text
1. 建立 ModelPool
2. 載入 M0.6 / M0.9 / M1.1 / M1.5 / M2.3
3. 加入 model cache hit/miss logging
4. 設計 scheduler config
5. 比較 resolution-only vs model-switching
```

## Phase 3：OS scheduling control

```text
1. 實作 CPU affinity mode
2. 實作 nice priority mode
3. 若權限允許，實作 chrt real-time mode
4. 跑 CPU stress 比較
5. 分析 context switch / P99 latency / deadline miss
```

## Phase 4：Queue / QoS

```text
1. 設計 FIFO / latest-only / deadline-drop
2. 掃 publisher rate
3. 掃 QoS depth
4. 分析 freshness_ms / dropped_frames / e2e latency
```

## Phase 5：Accuracy-latency Pareto

```text
1. 準備 100~500 張 labeled images
2. 跑 RepViT family + resolution sweep
3. 計算 top-1 / top-5
4. 畫 Pareto frontier
```

---

# 18. 總結

目前專題已經有完整的基礎：

```text
OS-aware runtime
RepViT / LLM workload
adaptive scheduler
admission control
stress testing
concurrent workload
ROS 2 pipeline
CPU/GPU comparison
RepViT family benchmark
```

下一步若想做得更有研究深度，建議不要再單純增加更多 latency benchmark，而是應聚焦：

```text
deadline miss decomposition
+
true model/device switching
+
OS scheduling control
+
queue freshness / QoS
+
accuracy-latency Pareto
```

最推薦的最終方向是：

> **EdgeAI-ROS Runtime：結合延遲分解、異質裝置調度與 OS 排程控制的 RepViT 即時推論系統**

這樣的專題不只是展示一個 AI model 能跑，而是能展示：

> AI inference performance is a full-stack OS problem.

這會比一般的 AI benchmark 或單純 ROS pipeline 更有研究深度，也更適合作業系統期末專題。
