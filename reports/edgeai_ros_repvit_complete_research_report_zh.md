# EdgeAI-ROS 中 RepViT 視覺推論與 LLM Runtime 自適應推論之完整研究報告

## 摘要

本研究以 EdgeAI-ROS 為實驗平台，探討 RepViT 視覺模型與本地大型語言模型（LLM）在 ROS 2 邊緣 AI pipeline 中的即時推論表現。研究重點包含六個面向：
- 第一，驗證 RepViT-M0.9 在不同 deadline 下的自適應解析度調整效果；
- 第二，比較 `predictive_adaptive`、`static_large` 與 `static_small` 三種 policy 的 latency 與 deadline miss；
- 第三，分析 CPU、GPU 與系統壓力對推論穩定性的影響；
- 第四，擴展至 RepViT model family，測試 RepViT-M0.6、M0.9、M1.0、M1.1、M1.5 與 M2.3 在相同設定下的效能差異。
- 第五，加入 deadline decomposition、true model/device switching、QoS/queue policy 與 CPU affinity 實驗，將本研究從單純 latency benchmark 延伸為 OS-aware runtime analysis。
- 第六，整合 Llama-3.2-1B 與 Gemma-3-1B-it 本地 LLM workload，評估 standalone LLM deadline behavior，以及 RepViT + LLM concurrent workload 對多模型邊緣系統的影響。

早期實驗使用 7 張固定真實 ImageNet 類型圖片作為 sanity-check workload；後續 accuracy-latency 與修改後 runtime 驗證則使用從 ImageNet validation split 建立的 1000-class controlled subset。透過 ROS 2 Humble pipeline 進行連續影像發布、模型推論與 latency logging 後，主要實驗結果顯示，在 RepViT-M0.9 上，GPU 可將 200 ms deadline 下的平均 inference latency 從 CPU 的 90.432 ms 降至 47.748 ms，deadline miss rate 從 1.7% 降至 0%。在 CPU mixed stress 下，deadline miss rate 上升至 29.4%，顯示 CPU-only edge inference 對系統壓力高度敏感。RepViT model family 實驗則顯示，CPU 上 M0.6、M0.9、M1.0 與 M1.1 較適合 200 ms deadline，而 M1.5 開始出現明顯 miss，M2.3 在 CPU 上 miss rate 達 74.8%；相對地，GPU 可讓所有模型大幅降低 latency，M2.3 的 GPU 平均 inference latency 為 74.516 ms，僅出現 0.7% deadline miss。

後續深化實驗顯示，GPU 將 RepViT-M0.9 static inference 平均時間由 CPU 的 84.633 ms 降至 38.047 ms，但 non-model overhead 從 14.660 ms 增至 16.691 ms，顯示加速模型後，ROS/Python preprocessing、message handling 與 runtime overhead 的相對重要性上升。在同樣 200 ms deadline、同樣 ImageNet subset、同樣 QoS/queue policy 下，重新執行並取 warm-up 後相同 76 筆 records 比較時，resolution-only GPU M0.9 的平均 inference 為 51.153 ms、0% miss；model/device switching 則穩定選擇 GPU M1.5，平均 inference 為 91.893 ms、0% miss，提供較高 quality operating point 但承擔較高 latency。QoS/queue stress 實驗則顯示 best-effort depth=1 搭配 deadline-drop 可降低 freshness tail，但會犧牲部分 frames 並出現少量 miss。

在依據研究深化建議修改程式後，本研究再次以 ImageNet 1000-class subset 真實影像執行推薦 runtime 組合。新版 pipeline 新增 publisher 端 timestamp、logger receive/write timing、model switch count、cache miss、GPU memory、CPU migration、arrival/service rate 與 utilization rho 等欄位。實測結果顯示，87 個 frames 中全部被 accept，僅第一個 M2.3 frame 因 cold-start/tail latency 發生 1 次 deadline miss；排除前 5 個 warm-up frames 後，RepViT-M1.5 GPU 的平均 inference 為 79.800 ms、P95 inference 為 111.489 ms、平均 E2E 為 99.096 ms，deadline miss rate 為 0%。此結果驗證修改後的 true model switching 與 admission control 可在 200 ms deadline 下穩定選擇較高品質的 GPU M1.5 operating point。

LLM 對齊實驗顯示，Llama-3.2-1B 在 standalone 200/150 ms per-token deadline 下可維持 0% miss rate，100 ms 開始進入臨界區；Gemma-3-1B-it 在 200 ms 下已接近能力邊界，none stress miss rate 為 16.7%，CPU stress 下升至 42.9%。在 RepViT + LLM concurrent workload 中，Llama 於 none/200 ms 下 LLM miss rate 為 0%，而 Gemma 在相同設定下為 71.4%；mixed stress + 200 ms 下 Gemma LLM miss rate 達 100%。此結果顯示，不同 1B 等級 LLM 即使參數量相近，其 runtime deadline behavior 仍可能差異極大。

整體而言，本研究證實在 ROS-based edge AI 系統中，模型大小、裝置選擇、OS pressure、自適應 policy、QoS/queue freshness、non-model overhead 與多模型 concurrent workload 會共同影響即時推論可靠性。對 200 ms 等級的視覺即時需求而言，RepViT-M0.9 搭配 GPU 是穩定基線；若追求更高 accuracy，需透過 device-aware model switching 與 queue/admission control 管理 tail latency。對本地 LLM workload 而言，Llama-3.2-1B 較適合作為即時 edge LLM baseline，而 Gemma-3-1B-it 需要更寬鬆 deadline、較積極 admission control 或推論後端最佳化。

## 1. 研究介紹

邊緣 AI 系統常需要在有限硬體資源下處理連續影像串流，例如機器人感知、行動裝置視覺、嵌入式監控與自動化檢測。此類應用不只要求模型準確率，也要求 inference latency 能穩定落在 deadline 內。若單次推論延遲過高，後續 ROS node 可能累積 queue、增加 end-to-end latency，甚至導致控制或決策延誤。

RepViT 是一系列面向 mobile/edge deployment 的輕量化視覺模型。其設計目標是在維持 ImageNet top-1 accuracy 的同時降低 latency。本研究將 RepViT 整合進 EdgeAI-ROS，以 ROS 2 pipeline 方式評估其在 CPU/GPU、不同 deadline、不同 policy、壓力負載與不同模型大小下的實際表現。進一步地，本研究也將 Llama-3.2-1B 與 Gemma-3-1B-it 兩個本地 LLM 加入同一 runtime framework，分析 token generation deadline、defer/admission control 與 vision-language concurrent workload 對系統可靠性的影響。

## 2. 研究動機

本研究的動機來自六個問題：

1. **模型準確率不等於系統可用性。** 即使模型在 ImageNet 上有良好 top-1 accuracy，部署到 ROS pipeline 後仍會受到資料傳輸、Python runtime、OS scheduling、CPU/GPU initialization 與 logging overhead 影響。
2. **即時系統需要 deadline-aware inference。** 邊緣系統常需要在固定時間內完成推論，因此平均 latency 之外，也必須觀察 P95 latency、max latency 與 deadline miss rate。
3. **自適應策略需要實測驗證。** Adaptive inference 的核心假設是，當系統壓力上升或 deadline 變嚴格時，可透過降低解析度或選擇較輕模型維持 deadline。然而這種策略是否有效，必須在同一批真實影像與相同 pipeline 中實測。
4. **模型推論時間不等於 pipeline latency。** 當 GPU 降低 inference latency 後，preprocess、message passing、queueing、logging 與 OS scheduling 可能成為新的瓶頸，因此需要 deadline decomposition。
5. **處理每一張 frame 不一定最適合 real-time perception。** 在高負載下，queue freshness、admission control、drop/defer 與 QoS policy 會影響實際可用性，因此需要從 queueing system 的角度分析 ROS 2 perception pipeline。
6. **邊緣 AI 正逐漸從單一視覺模型走向多模型 workload。** 實際機器人或 edge agent 可能同時執行視覺感知與語言推理，因此需評估 RepViT 與 LLM concurrent execution 對 GPU、CPU、deadline miss 與 admission control 的共同影響。

## 3. 文獻回顧

### 3.1 Edge AI 與即時推論

Edge AI 將模型部署於資料產生端附近，可降低網路傳輸延遲並提升隱私性。然而 edge device 的 CPU、GPU、記憶體與功耗有限，因此模型架構與 runtime scheduling 會直接影響 latency 與 deadline reliability。對機器人或互動式視覺系統而言，P95 latency 與 deadline miss rate 往往比單純平均 latency 更能反映部署風險。

### 3.2 RepViT

RepViT 以 lightweight CNN 為基礎，吸收 Vision Transformer 在結構設計上的優點，目標是在 mobile device 上取得較佳 accuracy-latency trade-off。官方模型包含 M0.6、M0.9、M1.0、M1.1、M1.5 與 M2.3。依官方資料，M0.9 至 M2.3 的 ImageNet top-1 accuracy 從 78.7% 提升至 83.3%，但參數量與 MACs 也隨之增加。本研究進一步測試這些模型在 ROS 2 pipeline 中的 CPU/GPU latency 與 deadline miss。

### 3.3 ROS 2 與邊緣 AI pipeline

ROS 2 提供分散式 node、topic 與 message passing 機制，適合建構機器人 perception pipeline。本研究使用三個主要 node：`image_publisher` 發布圖片、`adaptive_repvit_node` 執行模型推論與 policy decision、`edgeai_logger` 記錄 latency 與系統狀態。此架構可以觀察模型推論以外的 end-to-end overhead，因此比單獨 benchmark PyTorch forward pass 更接近實際部署情境。

### 3.4 Deadline-aware adaptive inference

Deadline-aware adaptive inference 透過 runtime decision 在不同品質層級間切換，例如降低解析度、使用小模型、切換 CPU/GPU device，或 defer/drop request。其目標是在資源壓力下維持 deadline compliance，同時避免處理過舊的 stale frames。本研究的 `predictive_adaptive` policy 會根據 latency prediction、deadline、runtime pressure 與 queue/admission decision 選擇 runtime config；`static_large` 與 `static_small` 則作為固定高品質與低品質 baseline。

### 3.5 Edge LLM runtime 與多模型 workload

近年的 edge AI 系統不再只執行單一視覺模型，也開始在本地執行小型 LLM，用於語意理解、任務規劃、摘要或人機互動。LLM 的 runtime behavior 與視覺模型不同：視覺模型通常以 frame-level latency 衡量，而 LLM 需要同時觀察 TTFT、TPOT、tokens/sec、output token budget 與 per-token deadline。當視覺模型與 LLM 共用 CPU/GPU、ROS executor、Docker runtime 與 memory bandwidth 時，即使兩者各自可在 standalone 條件下達標，concurrent workload 仍可能造成 tail latency 放大。因此，本研究將 Llama-3.2-1B 與 Gemma-3-1B-it 作為本地 LLM workload，分析不同 1B 等級模型在相同 EdgeAI-ROS pipeline 中的 deadline reliability。

## 4. 實驗平台與方法

### 4.1 系統環境

| 項目 | 設定 |
|---|---|
| 專案 | EdgeAI-ROS |
| Middleware | ROS 2 Humble |
| Container | `edgeai-ros:humble` |
| 主要模型 | RepViT-M0.9 |
| Model source | `official:/workspace/EdgeAI-ROS/RepViT` |
| Checkpoint | `checkpoints/repvit_m0_9_distill_300e.pth` |
| LLM models | Llama-3.2-1B, Gemma-3-1B-it |
| LLM paths | `Llama-3.2-1B`, `gemma-3-1b-it` |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| NVIDIA driver | 610.47 |
| CPU mode | Docker without `--gpus all` |
| GPU mode | Docker with `--gpus all` |

所有正式結果皆確認：

```text
fallback_model = 0
model_source = official:/workspace/EdgeAI-ROS/RepViT
```

這代表結果來自官方 RepViT checkpoint，而不是 fallback CNN。

### 4.2 ROS pipeline

本研究使用三類 pipeline。第一類為影像推論 pipeline：

```text
image_publisher -> adaptive_repvit_node -> edgeai_logger
```

`image_publisher` 以固定圖片資料夾循環發布影像。`adaptive_repvit_node` 依 policy 選擇推論設定並輸出 latency message。`edgeai_logger` 將每個 frame 的結果寫入 CSV，包含 `infer_ms`、`e2e_ms`、`deadline_ms`、`deadline_miss`、`level`、`image_size`、`action`、CPU 使用率、RSS memory、context switch 與 page fault 等欄位。

修改後的 pipeline 進一步在不新增 ROS message type 的前提下，透過 `Image.header.frame_id` 夾帶 publisher 端 metadata，包括 `publisher_seq`、`image_read_start_epoch_ms`、`image_read_end_epoch_ms` 與 `image_publish_epoch_ms`。`adaptive_repvit_node` 解析這些 timestamp 後，可計算 `image_load_ms`、`publish_overhead_ms`、`stale_frame_age_ms`、`freshness_ms`、`arrival_rate_hz`、`service_rate_hz` 與 `utilization_rho`。`edgeai_logger` 也新增 `logger_receive_ms` 與 `logging_ms`，用於觀察 CSV logging 對 tail latency 的影響。

第二類為 LLM-only pipeline：

```text
prompt_publisher -> adaptive_llm_node -> edgeai_logger
```

`prompt_publisher` 以固定 prompt period 發布文字請求。`adaptive_llm_node` 載入本地 Hugging Face causal LM，根據 deadline 與 pressure 選擇 context length、`max_new_tokens` 與 action，包括 `accept`、`degrade` 與 `defer`。LLM logger 記錄 `ttft_ms`、`tpot_ms`、`tokens_per_sec`、`output_tokens`、`deadline_miss`、`level`、`action`、CPU/memory 與 process-level OS metrics。

第三類為 RepViT + LLM concurrent pipeline：

```text
image_publisher -> adaptive_repvit_node -> edgeai_logger
prompt_publisher -> adaptive_llm_node   -> edgeai_logger
```

此 pipeline 同時發布影像與 prompt，用於觀察 vision workload 與 LLM workload 共用 GPU/CPU 時的 deadline interaction。RepViT 端固定使用 200 ms frame deadline；LLM 端則使用 200、150 與 100 ms/token deadline。

### 4.3 Warm-up 處理

早期 GPU 測試發現 CUDA lazy initialization 會使第一筆 inference latency 過高，影響 predictor 與 deadline decision。因此本研究在模型載入後加入 dummy forward warm-up：

```text
model load -> checkpoint load -> dummy forward -> cuda synchronize -> measured inference
```

此外，所有正式統計皆排除前 5 frames 作為 warm-up frames。

LLM 實驗則以 startup delay 避免 prompt 早於模型 ready。Llama-3.2-1B 使用 45 s startup delay；Gemma-3-1B-it 因模型載入與初始化較慢，使用 105 s startup delay。此設計使比較重點落在模型 ready 後的 runtime behavior，而不是模型載入時間。

### 4.4 Policy 與 level 設定

RepViT-M0.9 adaptive 實驗使用三個解析度層級：

| Level | Image size | Quality interpretation |
|---:|---:|---|
| 0 | 160 | low latency / lower quality |
| 1 | 192 | middle |
| 2 | 224 | high quality / higher latency |

測試 policy 包含：

| Policy | 說明 |
|---|---|
| `predictive_adaptive` | 根據 predicted latency、deadline 與 pressure 自適應選 level |
| `static_large` | 固定 level 2，解析度 224 |
| `static_small` | 固定 level 0，解析度 160 |

### 4.5 評估指標

| 指標 | 定義 |
|---|---|
| `infer_ms` | 模型推論時間 |
| `e2e_ms` | subscriber callback 內從 receive 到 result publish 的 end-to-end latency；logger write time 另以 `logging_ms` 記錄 |
| `deadline_miss` | 目前 RepViT runtime CSV 以 `infer_ms > deadline_ms` 判定的 model-runtime miss；E2E 與 freshness 另行分析 |
| `deadline_miss_rate` | deadline miss frames / total frames after warm-up |
| `P95 infer_ms` | inference latency 的第 95 百分位 |
| `avg_level` | 平均自適應品質層級 |
| `avg_image_size` | 平均輸入解析度 |
| `preprocess_ms` | image message 轉換、resize、normalize 與 tensor transfer 時間 |
| `postprocess_ms` | logits 轉換為 top-1 prediction 的時間 |
| `model_get_ms` | 從 model cache 取得模型的時間，含 cold load 時間 |
| `model_load_latency_ms` | cold-start 載入模型與 warm-up 的時間 |
| `model_cache_hit` | 模型是否來自 cache |
| `model_cache_miss` | 模型是否觸發 cache miss 或 cold path |
| `model_switched` | 本 frame 是否相較前一 frame 切換 runtime config |
| `model_switch_count` | 累積 model/device/config 切換次數 |
| `image_load_ms` | publisher 端影像讀取時間 |
| `publish_overhead_ms` | publisher publish 到 subscriber callback receive 的近似 middleware/queue delay |
| `logger_receive_ms` | result publish 到 logger callback receive 的近似延遲 |
| `logging_ms` | logger callback 內 CSV write/flush 近似時間 |
| `freshness_ms` | frame 從 publisher timestamp 到 inference callback 完成的近似新鮮度延遲 |
| `non_model_overhead_ms` | `e2e_ms - infer_ms`，代表非模型推論成本 |
| `arrival_rate_hz` / `service_rate_hz` | queueing analysis 中的到達率與服務率估計 |
| `utilization_rho` | `arrival_rate_hz / service_rate_hz`，用於判斷 queue 是否趨近過載 |
| `cpu_migration_delta` | process CPU core 是否在相鄰 sample 間發生 migration |
| `gpu_memory_allocated_mb` | PyTorch CUDA allocated memory |
| `ttft_ms` | LLM time-to-first-token 近似值 |
| `tpot_ms` | LLM 每 token 平均生成時間，作為 per-token deadline 的主要比較指標 |
| `tokens_per_sec` | LLM token generation throughput |
| `output_tokens` | LLM 實際輸出 token 數；`defer` row 通常為 0 |
| `deferred_requests` | LLM admission control 累積 defer 次數 |

## 5. Dataset

本研究依實驗目的使用三種影像資料集。早期實驗使用少量固定真實圖片作為可重複的 real-image workload；後續 accuracy-latency 與修改後 runtime 驗證則改用從 ImageNet validation split 建立的 1000-class controlled subset。因此，本研究的 dataset 不再只限於 7 張圖片。

### 5.1 七張真實圖片 sanity-check set

早期 deadline、policy 與 stress 實驗使用 7 張真實 ImageNet 類型圖片，位於：

```text
data/imagenet_labeled_images_resized
```

影像先經 resize，降低影像讀取與 ROS message transport 的干擾。資料集如下：

| Image | Semantic group | 主要觀察到的 ImageNet label |
|---|---|---|
| `apple.jpg` | apple | `Granny Smith` |
| `banana.jpg` | banana | `banana` |
| `cat.jpg` | cat | `tabby` / `Egyptian cat` |
| `dog.jpg` | dog | `Samoyed` |
| `espresso.jpg` | espresso | `espresso` |
| `goldfish.jpg` | goldfish | `goldfish` |
| `zebra.jpg` | zebra | `zebra` |

此資料集主要用於 sanity check 與早期 pipeline latency 測試，不用來估計正式 classification accuracy。

### 5.2 ImageNet 1000-class validation subset

為了讓 accuracy-latency 分析更接近 ImageNet 評估，本研究後續從：

```text
imagenet-object-localization-challenge.zip
```

抽取 validation images，建立每個 ImageNet synset/class 各 1 張的 balanced subset，共 1000 張圖片，位於：

```text
data/imagenet_val_subset_1_per_class/images
data/imagenet_val_subset_1_per_class/labels.csv
```

此 subset 搭配 `LOC_synset_mapping.txt` 建立 0-based ImageNet class index，可用於估計 RepViT model family 的 relative top-1 accuracy。雖然它仍不是完整 50,000 張 ImageNet validation set，但已比 7 張 sanity-check 圖片更適合比較 M0.6、M0.9、M1.0、M1.1、M1.5 與 M2.3 的 accuracy-latency trade-off。

修改後 runtime 驗證實驗也使用此 1000-class subset 作為連續 ROS image stream，以確認新增的 deadline decomposition、true model switching、queue/admission control 與 OS scheduling metrics 能在真實照片 workload 上正常運作。

### 5.3 Synthetic test images

少數 pipeline smoke test 使用 synthetic images，位於：

```text
data/test_images
```

這些圖片只用於快速確認 ROS node、CSV logger 與 Docker pipeline 可啟動，不納入正式 accuracy 或 latency 結論。

## 6. 實驗設計

### 6.1 Deadline experiment

使用 RepViT-M0.9 與 `predictive_adaptive` policy，在 CPU 與 GPU 上分別測試：

```text
250 ms, 200 ms, 150 ms
```

此實驗用於觀察 deadline 壓力增加時，adaptive policy 是否會降低解析度或產生 miss。

### 6.2 Policy comparison

固定 deadline 為 200 ms，比較：

```text
predictive_adaptive
static_large
static_small
```

此實驗用於比較 adaptive policy 與固定解析度 baseline 的 latency-quality trade-off。

### 6.3 Stress experiment

固定 deadline 為 200 ms，使用 `predictive_adaptive`，比較：

```text
no stress
cpu stress
mixed stress
```

此實驗用於觀察 OS pressure 對 CPU/GPU 推論穩定性的影響。

### 6.4 RepViT model family benchmark

為比較不同 RepViT 模型大小，在 CPU 與 GPU 上測試：

```text
RepViT-M0.6, M0.9, M1.0, M1.1, M1.5, M2.3
```

設定固定為：

```text
policy = static_large
image_size = 224
deadline = 200 ms
duration = 90 s per run
warmup_frames_excluded = 5
```

此設計刻意不使用 adaptive 切換，目的是隔離「模型大小」對 latency 與 deadline miss 的影響。

### 6.5 Research-depth runtime experiments

根據後續研究深化建議，本研究再加入四組實驗，使分析不只停留在 latency 數值，而能回答 latency 來源與 OS/runtime mechanism 的影響：

| 實驗組 | 設定 | 目的 |
|---|---|---|
| Deadline decomposition | RepViT-M0.9 static large，CPU vs GPU | 分解 preprocess、inference、postprocess 與 non-model overhead |
| Resolution-only vs model/device switching | GPU resolution-only M0.9 vs M0.6/M0.9/M1.1/M1.5/M2.3 CPU/GPU config | 比較只調解析度與真正切換模型/裝置 |
| QoS / queue policy under stress | CPU stress 下比較 reliable FIFO 與 best-effort depth=1 + deadline-drop | 觀察 freshness 與 tail latency |
| CPU affinity under stress | CPU-only RepViT-M0.9，default vs restricted CPU affinity | 測試 OS-level scheduling control 是否改善或傷害 deadline behavior |

新增程式支援包含 `(model, image_size, device, quality_score)` runtime config、以 `(model, device)` 為 key 的 model cache、`qos_depth`、`qos_reliability`、`queue_policy`、`cpu_affinity`、`nice_delta` 與 `preload_models`。

### 6.6 修改後 runtime 驗證實驗

在完成 deadline decomposition、true model switching、CPU scheduling metrics 與 queueing/admission control 的程式調整後，本研究使用報告中最推薦的 runtime 組合重新執行一次 end-to-end pipeline：

| 項目 | 設定 |
|---|---|
| Dataset | `data/imagenet_val_subset_1_per_class/images` |
| Policy | `predictive_adaptive` |
| Deadline | 200 ms |
| Models | M0.6, M0.9, M1.1, M1.5, M2.3 |
| Image sizes | 160, 192, 224, 224, 224 |
| Devices | CPU, CPU, GPU, GPU, GPU |
| QoS | best-effort, depth=1 |
| Queue policy | `deadline_drop` |
| Model loading | `preload_models=true` |
| Admission control | `drop_stale_ms=500.0`, `overload_rho=1.0` |
| Duration | `timeout 75s` controlled run |

此實驗目的不是重新測所有 baseline，而是驗證修改後的 runtime 是否能同時正確輸出 decomposition、model switching、queue/admission 與 OS scheduling 指標，並確認新 admission control 不會在 cold-start 階段錯誤地丟棄所有 frames。

### 6.7 LLM standalone 與 RepViT+LLM concurrent experiments

為了將研究範圍從 vision-only runtime 擴展至 vision-language multi-workload，本研究加入 Llama-3.2-1B 與 Gemma-3-1B-it 的對齊實驗。

LLM-only standalone matrix 設定如下：

| 項目 | 設定 |
|---|---|
| Models | Llama-3.2-1B, Gemma-3-1B-it |
| Policy | `predictive_adaptive` |
| Device | CUDA |
| Deadline | 200, 150, 100, 80 ms/token |
| Stress | none, CPU stress |
| Prompt period | 15 sec |
| Llama dtype | `float16` |
| Gemma dtype | `bfloat16` |
| Duration | 120 sec per run |

RepViT + LLM concurrent matrix 設定如下：

| 項目 | 設定 |
|---|---|
| Vision model | RepViT-M0.9 |
| LLM models | Llama-3.2-1B, Gemma-3-1B-it |
| RepViT deadline | 200 ms |
| LLM deadline | 200, 150, 100 ms/token |
| Stress | none, mixed stress |
| Image period | 0.5 sec |
| Prompt period | 15 sec |
| Duration | 120 sec per run |

Gemma 3 需要較新的 Hugging Face Transformers 支援，因此 Gemma 測試在容器內暫時安裝 `transformers==4.50.0`；Llama 則使用 `transformers==4.45.2`。

## 7. 實驗結果與分析

### 7.1 RepViT-M0.9 CPU/GPU deadline 結果

| Device | Deadline | Rows | Misses | Miss rate | Avg infer ms | Median infer ms | P95 infer ms | Avg level | Avg image size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU | 250 | 124 | 1 | 0.008 | 88.375 | 71.889 | 161.600 | 1.258 | 200.258 |
| CPU | 200 | 119 | 2 | 0.017 | 90.432 | 79.991 | 145.678 | 1.983 | 223.462 |
| CPU | 150 | 120 | 5 | 0.042 | 77.733 | 67.786 | 145.736 | 1.167 | 197.333 |
| GPU | 250 | 107 | 0 | 0.000 | 72.288 | 67.496 | 114.352 | 2.000 | 224.000 |
| GPU | 200 | 114 | 0 | 0.000 | 47.748 | 40.410 | 91.165 | 2.000 | 224.000 |
| GPU | 150 | 116 | 0 | 0.000 | 54.576 | 47.378 | 109.072 | 2.000 | 224.000 |

**分析。** CPU 在 deadline 變嚴格時會出現較多 miss，其中 150 ms miss rate 為 4.2%。GPU 在三種 deadline 下皆維持 0% miss rate，且 average level 皆為 2，代表 GPU 有足夠資源維持 224 解析度而不需降級。CPU 的 `avg_level` 在 150 ms 降至 1.167，顯示 adaptive policy 確實有降低解析度以回應 deadline pressure。

### 7.2 Policy comparison

| Device | Policy | Deadline | Rows | Misses | Miss rate | Avg infer ms | P95 infer ms | Avg level | Avg image size |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU | predictive_adaptive | 200 | 119 | 2 | 0.017 | 90.432 | 145.678 | 1.983 | 223.462 |
| CPU | static_large | 200 | 119 | 1 | 0.008 | 86.892 | 156.295 | 2.000 | 224.000 |
| CPU | static_small | 200 | 124 | 5 | 0.040 | 85.303 | 170.185 | 0.000 | 160.000 |
| GPU | predictive_adaptive | 200 | 114 | 0 | 0.000 | 47.748 | 91.165 | 2.000 | 224.000 |
| GPU | static_large | 200 | 115 | 0 | 0.000 | 44.600 | 77.787 | 2.000 | 224.000 |
| GPU | static_small | 200 | 113 | 0 | 0.000 | 52.345 | 107.321 | 0.000 | 160.000 |

**分析。** 在無 stress 的 200 ms 設定下，RepViT-M0.9 本身已能大多數時間維持 deadline，因此 `static_large` 與 `predictive_adaptive` 差距不大。CPU 上 `static_small` 未必更穩，可能因 pipeline overhead、影像處理與 scheduling jitter 使低解析度優勢被部分抵消。GPU 上三種 policy 皆無 miss，表示 GPU 運算資源足以吸收不同 policy 的差異。

### 7.3 Stress experiment

| Device | Condition | Deadline | Rows | Misses | Miss rate | Avg infer ms | P95 infer ms | Avg level | Avg image size |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU | no stress | 200 | 119 | 2 | 0.017 | 90.432 | 145.678 | 1.983 | 223.462 |
| CPU | cpu stress | 200 | 110 | 6 | 0.055 | 109.901 | 213.595 | 1.436 | 205.964 |
| CPU | mixed stress | 200 | 85 | 25 | 0.294 | 192.041 | 415.654 | 0.200 | 166.400 |
| GPU | no stress | 200 | 114 | 0 | 0.000 | 47.748 | 91.165 | 2.000 | 224.000 |
| GPU | cpu stress | 200 | 96 | 0 | 0.000 | 52.223 | 101.399 | 2.000 | 224.000 |
| GPU | mixed stress | 200 | 105 | 0 | 0.000 | 68.544 | 139.326 | 2.000 | 224.000 |

**分析。** CPU 在 mixed stress 下 miss rate 達 29.4%，P95 inference latency 達 415.654 ms，表示 CPU-only deployment 容易受到背景負載影響。Adaptive policy 在 mixed stress 下將 `avg_level` 降至 0.200、`avg_image_size` 降至 166.400，但仍無法完全避免 deadline miss。GPU 在 stress 下仍維持 0% miss rate，顯示將主要推論負載移至 GPU 可有效隔離部分 CPU pressure。

### 7.4 較長時間 adaptive follow-up 結果

| Experiment | Policy | Deadline | Rows | Misses | Miss rate | Avg infer ms | P95 infer ms | Avg level | Avg image size |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| long 250 ms | predictive_adaptive | 250 | 328 | 0 | 0.000 | 84.291 | 131.285 | 2.000 | 224.000 |
| long 200 ms | predictive_adaptive | 200 | 339 | 3 | 0.009 | 78.020 | 125.252 | 1.147 | 196.720 |
| long 150 ms | predictive_adaptive | 150 | 339 | 4 | 0.012 | 78.702 | 115.577 | 1.988 | 223.622 |
| static large | static_large | 200 | 220 | 0 | 0.000 | 78.137 | 117.315 | 2.000 | 224.000 |
| static small | static_small | 200 | 219 | 1 | 0.005 | 60.502 | 100.228 | 0.000 | 160.000 |
| cpu stress | predictive_adaptive | 200 | 212 | 4 | 0.019 | 99.637 | 171.214 | 1.981 | 223.396 |
| mixed stress | predictive_adaptive | 200 | 201 | 1 | 0.005 | 98.723 | 150.710 | 0.000 | 160.000 |

**分析。** 較長時間 follow-up 顯示，在同一組真實圖片上，adaptive policy 能在不同 deadline 下維持低 miss rate。這組結果與最終 CPU/GPU stress 結果方向一致，但數值存在差異，可能與單次 run 的 OS 狀態、Docker resource scheduling 與背景負載有關。因此本研究在結論上以多組實驗共同趨勢為主，而非僅解讀單一 run。

### 7.5 RepViT model family CPU 結果

| Model | Params M | Official top-1 300e | MACs G | Rows | Misses | Miss rate | Avg infer ms | P95 infer ms | Max infer ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0.6 | 2.487 | 74.1 | N/A | 155 | 0 | 0.000 | 51.770 | 78.318 | 122.425 |
| M0.9 | 5.104 | 78.7 | 0.8 | 158 | 0 | 0.000 | 76.683 | 106.702 | 133.679 |
| M1.0 | 6.853 | 80.0 | 1.1 | 158 | 3 | 0.019 | 97.475 | 148.712 | 282.804 |
| M1.1 | 8.289 | 80.7 | 1.3 | 157 | 1 | 0.006 | 93.889 | 143.766 | 220.641 |
| M1.5 | 14.130 | 82.3 | 2.3 | 157 | 13 | 0.083 | 149.813 | 212.651 | 287.491 |
| M2.3 | 23.047 | 83.3 | 4.5 | 155 | 116 | 0.748 | 231.901 | 324.443 | 497.657 |

**分析。** CPU 上的結果呈現明確模型大小效應。M0.6 與 M0.9 無 deadline miss；M1.0 與 M1.1 miss rate 仍低於 2%；M1.5 miss rate 上升至 8.3%；M2.3 平均 inference latency 已超過 200 ms，miss rate 達 74.8%。若部署環境只能使用 CPU，M0.9 是較佳 accuracy-latency 折衷，M1.0/M1.1 則需要更保守的 deadline 或 adaptive policy。

### 7.6 RepViT model family GPU 結果

| Model | Params M | Official top-1 300e | MACs G | Rows | Misses | Miss rate | Avg infer ms | P95 infer ms | Max infer ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0.6 | 2.487 | 74.1 | N/A | 155 | 0 | 0.000 | 24.943 | 40.547 | 51.716 |
| M0.9 | 5.104 | 78.7 | 0.8 | 157 | 0 | 0.000 | 36.299 | 56.678 | 69.463 |
| M1.0 | 6.853 | 80.0 | 1.1 | 150 | 0 | 0.000 | 36.567 | 54.920 | 74.359 |
| M1.1 | 8.289 | 80.7 | 1.3 | 156 | 0 | 0.000 | 34.411 | 56.986 | 73.449 |
| M1.5 | 14.130 | 82.3 | 2.3 | 155 | 0 | 0.000 | 55.230 | 91.661 | 107.434 |
| M2.3 | 23.047 | 83.3 | 4.5 | 153 | 1 | 0.007 | 74.516 | 114.775 | 203.806 |

**分析。** GPU 大幅改善所有模型的 latency。M2.3 在 CPU 上平均 231.901 ms，但 GPU 上降至 74.516 ms。這代表若硬體允許 GPU inference，較大模型可在維持 200 ms deadline 的情況下取得更高官方 ImageNet top-1 accuracy。

### 7.7 CPU/GPU speedup

| Model | CPU avg infer ms | GPU avg infer ms | Speedup |
|---|---:|---:|---:|
| M0.6 | 51.770 | 24.943 | 2.076x |
| M0.9 | 76.683 | 36.299 | 2.113x |
| M1.0 | 97.475 | 36.567 | 2.666x |
| M1.1 | 93.889 | 34.411 | 2.728x |
| M1.5 | 149.813 | 55.230 | 2.713x |
| M2.3 | 231.901 | 74.516 | 3.112x |

**分析。** GPU 對較大模型的效益更明顯。M2.3 speedup 達 3.112x，表示 GPU 對高運算量模型更能發揮 parallelism。

### 7.8 分類結果觀察

大部分圖片在不同 policy、deadline 與裝置下的 top-1 label 穩定，例如 banana、espresso、goldfish 與 zebra 幾乎皆維持相同 label。`cat.jpg` 偶爾在 `tabby` 與 `Egyptian cat` 間切換，屬於 ImageNet 細粒度類別相近造成的 top-1 變化。RepViT-M0.6 對 `apple.jpg` 預測為 `pomegranate`，而 M0.9 以上多預測為 `Granny Smith`，顯示較小模型可能在細節辨識上較弱。

此小節反映早期 7 張真實圖片 sanity-check set 的 qualitative observation；正式 accuracy-latency 比較已於下一節改用 ImageNet 1000-class validation subset。因此，7 張圖片結果只用來說明模型輸出是否大致合理，不作為正式 accuracy benchmark。

### 7.9 ImageNet 1000-class subset accuracy-latency 結果

為補足 7 張圖片資料集無法估計 classification accuracy 的限制，本研究從 `imagenet-object-localization-challenge.zip` 的 ILSVRC validation split 中，每個 synset/class 抽 1 張，共 1000 張，建立 balanced validation subset。所有圖片 resize 至 max side 512，並使用官方 `LOC_synset_mapping.txt` 對齊 0-based ImageNet class index。

| Model | Official top-1 300e | Subset top-1 | Correct | Avg infer ms | P95 infer ms | Deadline miss |
|---|---:|---:|---:|---:|---:|---:|
| M0.6 | 74.1 | 74.4 | 744 / 1000 | 27.048 | 50.422 | 0 |
| M0.9 | 78.7 | 78.4 | 784 / 1000 | 35.034 | 57.980 | 0 |
| M1.0 | 80.0 | 80.7 | 807 / 1000 | 36.376 | 59.540 | 0 |
| M1.1 | 80.7 | 81.6 | 816 / 1000 | 35.012 | 57.918 | 0 |
| M1.5 | 82.3 | 82.7 | 827 / 1000 | 70.244 | 143.316 | 11 |
| M2.3 | 83.3 | 83.7 | 837 / 1000 | 70.491 | 128.039 | 9 |

**分析。** 1000-class subset 的 top-1 accuracy 與 RepViT 官方 300e top-1 非常接近，M0.9 為 78.4% 對官方 78.7%，M2.3 為 83.7% 對官方 83.3%。這表示本研究的 checkpoint、label mapping 與 preprocessing 基本可信。Latency 方面，M0.6 至 M1.1 在 GPU 上可穩定維持 200 ms deadline；M1.5/M2.3 accuracy 較高，但在 0.1 s publish period 的 ROS pipeline 中出現少量 miss，顯示大模型的 tail latency 仍需 queue/admission control。

### 7.10 Deadline decomposition 結果

| Experiment | Rows | Misses | Avg preprocess ms | Avg infer ms | P95 infer ms | Avg non-model ms | Avg E2E ms | P95 E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU static M0.9 | 275 | 0 | 3.642 | 84.633 | 116.671 | 14.660 | 99.293 | 132.654 |
| GPU static M0.9 | 296 | 0 | 5.235 | 38.047 | 60.549 | 16.691 | 54.738 | 79.337 |

**分析。** GPU 將 RepViT-M0.9 static-large inference latency 由 84.633 ms 降至 38.047 ms，約 2.22x speedup。可是 non-model overhead 從 14.660 ms 上升至 16.691 ms，且 preprocess 也從 3.642 ms 上升至 5.235 ms，主要原因可能包含 tensor transfer 與 CUDA synchronization。換言之，GPU 使模型推論不再是唯一主導成本，ROS/Python pipeline、preprocessing 與 runtime overhead 的比例變得更重要。

### 7.11 Resolution-only adaptive 與 true model/device switching

原先版本曾將 150 ms 的 resolution-only run 與 200 ms 的 model/device switching run 放在同一張表中比較；此比較會混入 deadline 設定差異。為修正此問題，本研究重新執行兩組實驗，皆採用 200 ms deadline、ImageNet 1000-class subset、best-effort depth=1、`deadline_drop` queue policy、`preload_models=true`，並在 publisher 啟動前加入 `STARTUP_DELAY_SEC=35`，讓 inference node 先完成模型預載。由於兩次 run 的 publisher throughput 仍有差異，最終比較取兩組 warm-up 後共同可用的前 76 筆 records，形成 matched comparison。

| Experiment | Deadline | Matched rows | Selected models | Devices | Misses | Avg infer ms | P95 infer ms | Avg non-model ms | Avg E2E ms | P95 E2E ms | Avg freshness ms | P95 freshness ms |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Resolution-only GPU M0.9 | 200 | 76 | M0.9 | GPU | 0 | 51.153 | 96.136 | 18.748 | 69.901 | 120.282 | 198.812 | 286.457 |
| Model/device switching | 200 | 76 | M1.5 | GPU | 0 | 91.893 | 151.322 | 19.576 | 111.469 | 177.467 | 253.571 | 362.608 |

**分析。** 在相同 200 ms 設定與相同 matched rows 下，resolution-only GPU M0.9 明顯較快，平均 inference 為 51.153 ms，平均 E2E 為 69.901 ms；model/device switching 則穩定選擇 GPU M1.5，平均 inference 為 91.893 ms，平均 E2E 為 111.469 ms。兩者皆為 0% miss，代表在 200 ms deadline 下二者都能滿足即時需求。差異在於 operating point：resolution-only M0.9 提供較低 latency 與較大 deadline margin；model/device switching 則以較高 latency 換取較大模型的 quality proxy。此結果更精準地支持「model/device switching 不是單純更快，而是提供可控的 quality-latency trade-off」。

補充觀察 full-run 統計：resolution-only rerun 共有 81 筆 records，warm-up 後 76 筆；model/device switching rerun 共有 250 筆 records，warm-up 後主要為 M1.5，共 243 筆 M1.5 records，M1.5 full-run warm-up 後 miss rate 為 0.8%。由於 publisher image loading 與 timer scheduling 仍會造成 run-to-run throughput 差異，matched comparison 較適合用於本節的公平比較；嚴格統計仍需要更長時間與多次重複實驗。

### 7.12 QoS / queue policy under CPU stress

| Experiment | Rows | Accept | Drop | Misses | Miss rate | Avg infer ms | P95 infer ms | Avg freshness ms | P95 freshness ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Reliable FIFO stress | 96 | 96 | 0 | 0 | 0.000 | 92.771 | 144.489 | 791.109 | 3098.987 |
| Best-effort depth=1 + deadline-drop | 62 | 61 | 1 | 2 | 0.032 | 80.270 | 151.337 | 377.347 | 594.546 |

**分析。** Reliable FIFO 在 CPU stress 下保持所有已記錄 frames 都被處理，且沒有 deadline miss，但 freshness tail 很高，P95 freshness 達 3098.987 ms，表示系統可能處理到較舊的 frames。Best-effort depth=1 搭配 deadline-drop 則將 P95 freshness 降至 594.546 ms，但犧牲部分 frame 並出現 3.2% miss rate。這支持 real-time perception 的核心觀點：在強調即時性的場景中，處理最新 frame 往往比處理每一個 frame 更重要。

### 7.13 CPU affinity / priority control under CPU stress

| Experiment | Rows | Accept | Defer | Misses | Avg infer ms | P95 infer ms | Avg E2E ms | P95 E2E ms | P95 freshness ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU stress default preload | 201 | 201 | 0 | 0 | 104.145 | 158.725 | 120.190 | 175.993 | 355.092 |
| CPU stress restricted affinity preload | 214 | 0 | 214 | 0 | N/A | N/A | N/A | N/A | 269.143 |

**分析。** 在 CPU stress 下，預先載入模型後的 default scheduling 可維持 0% miss，平均 inference 為 104.145 ms。相反地，將 inference process 限制在 cores 2,3 後，第一個 frame 的高延遲使 admission controller 判定後續 frames 不可行，導致連續 defer。此結果不代表 CPU affinity 必然有害，而是說明「過度限制 CPU core」可能造成 service rate 低於 arrival rate，使 admission control 進入保守模式。未來若要使用 affinity，應搭配 stress process 的 cpuset 隔離，而不是只限制 inference node。

### 7.14 修改後 runtime 驗證結果

依據第 6.6 節設定，本研究以修改後的 pipeline 重新執行推薦 runtime 組合。第一次執行時發現 `DROP_STALE_MS=500` 會被 ROS 2 parameter system 視為 integer，而 node 端宣告為 double，造成 `drop_stale_ms` 型別不符。此問題已透過將 script 預設值改為 `0.0`，並於實驗中使用 `500.0` 解決。第二次執行時則發現 `deadline_drop` 在尚未有 latency sample 時過度保守，會因初始化預測值而將所有 frames drop。後續將 admission 流程改為先由 scheduler 估計候選 config，再以該 config 的 predicted latency 進行 admission decision，修正後可正常 accept 並執行推論。

修正後的完整 run 產生 87 筆 frame-level records，全部被 accept，沒有 drop 或 defer。整體僅第一筆 M2.3 GPU inference 發生 deadline miss；此筆 inference 為 246.811 ms、E2E 為 292.713 ms。scheduler 隨後先切至 M0.6 CPU，再穩定切至 M1.5 GPU。排除前 5 個 warm-up frames 後，統計結果如下：

| Metric | Value |
|---|---:|
| Rows after warm-up | 82 |
| Accepted frames | 82 |
| Dropped frames | 0 |
| Deferred frames | 0 |
| Deadline misses | 0 |
| Deadline miss rate | 0.000 |
| Selected model after warm-up | RepViT-M1.5 |
| Device | GPU |
| Avg image load ms | 20.033 |
| Avg publish overhead ms | 94.606 |
| Avg preprocess ms | 6.448 |
| Avg inference ms | 79.800 |
| P95 inference ms | 111.489 |
| Avg postprocess ms | 0.398 |
| Avg non-model overhead ms | 19.297 |
| P95 non-model overhead ms | 25.257 |
| Avg E2E ms | 99.096 |
| P95 E2E ms | 130.844 |
| Avg freshness ms | 213.791 |
| P95 freshness ms | 317.862 |
| Avg utilization rho | 0.284 |
| Avg effective FPS | 10.482 |
| CPU migrations | 31 |
| Avg GPU allocated memory MB | 323.097 |
| Avg GPU reserved memory MB | 436.000 |

前 3 筆 frame 的切換序列如下：

| Frame | Action | Model | Device | Infer ms | E2E ms | Deadline miss | Model switched | Switch count |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | accept | RepViT-M2.3 | GPU | 246.811 | 292.713 | 1 | 0 | 0 |
| 2 | accept | RepViT-M0.6 | CPU | 98.387 | 115.152 | 0 | 1 | 1 |
| 3 | accept | RepViT-M1.5 | GPU | 84.656 | 104.355 | 0 | 1 | 2 |

**分析。** 此結果驗證修改後的 true model switching pipeline 可正確回應 deadline miss：第一筆 M2.3 miss 後，系統短暫切至較小的 M0.6，接著選擇兼具 quality 與 latency 的 M1.5 GPU。warm-up 後 M1.5 GPU 的平均 inference 為 79.800 ms，P95 inference 為 111.489 ms，距離 200 ms deadline 仍有明顯餘裕。平均 E2E 為 99.096 ms，表示 end-to-end pipeline 在 200 ms deadline 下穩定；但平均 freshness 達 213.791 ms，P95 freshness 為 317.862 ms，說明即使推論本身不 miss，publisher 端 image loading、middleware delay 與 callback 排程仍會使 frame freshness 高於純 E2E inference latency。因此後續若要服務控制迴路，freshness 需與 deadline miss 並列為主要指標。

此 run 也顯示新增欄位能支援更細的 OS/runtime 分析。平均 `publish_overhead_ms` 為 94.606 ms，明顯高於 logger receive 與 logging cost，代表影像讀取、publisher timer、DDS queue 與 subscriber scheduling 是 freshness 的主要來源之一。平均 `utilization_rho` 為 0.284，表示服務率高於到達率，queue 不處於過載狀態；同時仍觀察到 31 次 CPU migration，顯示即使 GPU 負責主要推論，process-level scheduling jitter 仍可能影響 tail latency。

### 7.15 Llama-3.2-1B 與 Gemma-3-1B-it standalone 結果

| Model | Stress | Deadline | Rows | Avg TPOT | P95 TPOT | Tokens/sec | Miss rate | Actions |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Llama | none | 200 | 6 | 98.961 | 114.138 | 10.187 | 0.000 | accept:5; degrade:1 |
| Llama | none | 150 | 6 | 98.000 | 112.608 | 10.282 | 0.000 | accept:5; degrade:1 |
| Llama | none | 100 | 7 | 97.289 | 103.911 | 10.294 | 0.143 | degrade:7 |
| Llama | none | 80 | 7 | 99.795 | 109.824 | 10.054 | 1.000 | degrade:7 |
| Llama | cpu | 200 | 6 | 101.204 | 113.458 | 9.932 | 0.000 | accept:5; degrade:1 |
| Llama | cpu | 150 | 6 | 100.733 | 114.984 | 9.995 | 0.000 | accept:5; degrade:1 |
| Llama | cpu | 100 | 7 | 103.006 | 114.941 | 9.752 | 0.571 | degrade:7 |
| Llama | cpu | 80 | 7 | 102.239 | 113.290 | 9.818 | 1.000 | degrade:7 |
| Gemma | none | 200 | 6 | 162.936 | 209.434 | 6.302 | 0.167 | degrade:3; accept:3 |
| Gemma | none | 150 | 7 | 163.904 | 206.613 | 6.263 | 0.571 | degrade:7 |
| Gemma | none | 100 | 7 | 173.995 | 217.114 | 5.952 | 1.000 | degrade:7 |
| Gemma | none | 80 | 7 | 28.387 | 139.098 | 0.719 | 0.143 | defer:6; degrade:1 |
| Gemma | cpu | 200 | 7 | 202.113 | 246.234 | 5.060 | 0.429 | degrade:7 |
| Gemma | cpu | 150 | 7 | 193.560 | 224.933 | 5.216 | 1.000 | degrade:7 |
| Gemma | cpu | 100 | 7 | 193.539 | 223.992 | 5.237 | 1.000 | degrade:7 |
| Gemma | cpu | 80 | 7 | 59.053 | 216.943 | 1.404 | 0.286 | defer:5; degrade:2 |

**分析。** Llama-3.2-1B 的 standalone TPOT 約落在 98 至 103 ms/token，200 與 150 ms deadline 均可穩定達標；100 ms 是臨界點，CPU stress 下 miss rate 上升至 57.1%；80 ms 則全 miss。Gemma-3-1B-it 在正常生成時 TPOT 約落在 163 至 202 ms/token，明顯慢於 Llama。Gemma 的 200 ms 已是邊界條件，none stress 下 miss rate 為 16.7%，CPU stress 下升至 42.9%。

Gemma 在 80 ms 組出現大量 `defer`，因此平均 TPOT 下降不可解讀為模型變快。排除 defer 後，Gemma none/80ms 的生成型 row TPOT 為 198.711 ms/token，cpu/80ms 的生成型 rows 平均 TPOT 為 206.687 ms/token，且生成型 rows 都 miss。這代表 admission control 確實避免了不可能達標的請求，但不表示 Gemma 能在 80 ms/token 下穩定推論。

### 7.16 RepViT + LLM concurrent 結果

LLM 端結果如下：

| Model | Stress | LLM deadline | Rows | Avg TPOT | P95 TPOT | Tokens/sec | Miss rate | Actions |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Llama | none | 200 | 6 | 154.017 | 195.645 | 6.746 | 0.000 | accept:4; degrade:2 |
| Llama | none | 150 | 6 | 133.559 | 151.663 | 7.599 | 0.167 | degrade:4; accept:2 |
| Llama | none | 100 | 7 | 129.411 | 146.865 | 7.823 | 1.000 | degrade:7 |
| Llama | mixed | 200 | 5 | 154.272 | 199.370 | 6.678 | 0.200 | degrade:3; accept:2 |
| Llama | mixed | 150 | 7 | 156.739 | 181.878 | 6.449 | 0.571 | degrade:7 |
| Llama | mixed | 100 | 7 | 168.236 | 324.435 | 3.266 | 0.714 | degrade:5; defer:2 |
| Gemma | none | 200 | 7 | 211.380 | 257.510 | 4.802 | 0.714 | degrade:7 |
| Gemma | none | 150 | 7 | 224.070 | 251.427 | 4.505 | 1.000 | degrade:7 |
| Gemma | none | 100 | 7 | 37.473 | 183.616 | 0.545 | 0.143 | defer:6; degrade:1 |
| Gemma | mixed | 200 | 6 | 396.228 | 413.157 | 2.530 | 1.000 | degrade:6 |
| Gemma | mixed | 150 | 7 | 58.434 | 286.324 | 0.349 | 0.143 | defer:6; degrade:1 |
| Gemma | mixed | 100 | 7 | 55.451 | 271.710 | 0.368 | 0.143 | defer:6; degrade:1 |

RepViT 端結果摘要如下：

| LLM model | Stress | LLM deadline | Rows | RepViT miss rate | Avg infer | P95 infer | Avg E2E | P95 freshness |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Llama | none | 200 | 198 | 0.010 | 85.213 | 136.896 | 108.612 | 1906.594 |
| Llama | none | 150 | 216 | 0.014 | 61.239 | 107.994 | 79.507 | 847.420 |
| Llama | none | 100 | 222 | 0.009 | 58.372 | 100.033 | 79.080 | 680.640 |
| Gemma | none | 200 | 226 | 0.009 | 56.246 | 89.274 | 77.690 | 646.332 |
| Gemma | none | 150 | 223 | 0.013 | 59.959 | 103.546 | 81.700 | 1083.905 |
| Gemma | none | 100 | 229 | 0.000 | 48.774 | 84.146 | 68.663 | 548.546 |
| Llama | mixed | 200 | 167 | 0.018 | 98.093 | 167.183 | 120.174 | 3446.181 |
| Llama | mixed | 150 | 179 | 0.017 | 95.193 | 165.208 | 115.870 | 2474.391 |
| Llama | mixed | 100 | 129 | 0.093 | 137.201 | 254.919 | 164.605 | 3454.266 |
| Gemma | mixed | 200 | 166 | 0.012 | 96.670 | 163.522 | 123.848 | 2107.077 |
| Gemma | mixed | 150 | 201 | 0.000 | 77.936 | 131.955 | 100.651 | 1493.853 |
| Gemma | mixed | 100 | 202 | 0.005 | 78.906 | 133.887 | 101.542 | 2467.106 |

**分析。** Llama 在 concurrent none/200ms 下仍可達成 0% LLM miss rate，但 Gemma 在相同設定下 LLM miss rate 為 71.4%。在 mixed stress 下，Gemma 的 LLM 端壓力進一步放大：mixed/200ms 的平均 TPOT 為 396.228 ms/token，miss rate 為 100%。Gemma concurrent 的 100ms 與 mixed 150/100ms 出現大量 `defer`，因此總體 miss rate 看似下降到 14.3%；但排除 defer 後，這些生成型請求仍全部 miss。這些結果說明，低 deadline 下的 defer 是 admission control 的保護行為，不是模型達標。

RepViT 端在 concurrent 實驗中多數仍可維持低 miss rate，表示 vision pipeline 的 200 ms deadline 在 GPU 上有一定餘裕；但 mixed stress 會提高 RepViT inference、E2E 與 freshness tail。特別是 Llama mixed/100ms 下 RepViT miss rate 升至 9.3%，代表 concurrent LLM 與 stress workload 的組合仍可能傷害 vision deadline。

## 8. 綜合討論

### 8.1 Deadline reliability

若以 200 ms deadline 為目標，RepViT-M0.9 在 CPU 上大多可行，但仍會因 OS pressure 產生 miss。GPU 則可穩定維持 deadline，且不需降低解析度。進一步的 deadline decomposition 實驗顯示，M0.9 從 CPU 移到 GPU 後，平均 inference latency 由 84.633 ms 降至 38.047 ms，而 end-to-end latency 由 99.293 ms 降至 54.738 ms。這說明在 ROS pipeline 中，GPU 不只降低 forward pass 的平均延遲，也能增加 deadline margin 並降低 tail latency 風險。

### 8.2 Adaptive policy 的效果

Adaptive policy 在 CPU deadline pressure 或 mixed stress 下會降低 average level 與 image size，表示 policy 確實有回應壓力。但當系統負載過高時，例如 CPU mixed stress，僅靠降解析度仍不足以完全避免 miss。新增的 model/device switching 實驗顯示，若 policy 可同時選擇模型、解析度與 device，系統能在 200 ms deadline 下偏好 GPU M1.5，並維持 0% miss rate。修改後 runtime 驗證也顯示，系統在第一筆 M2.3 miss 後會先降到 M0.6，再收斂到 M1.5 GPU；排除 warm-up 後，M1.5 GPU 的平均 inference 為 79.800 ms、P95 inference 為 111.489 ms，deadline miss rate 為 0%。這代表 adaptive policy 的有效性高度取決於可調整維度是否足夠，而不只是單一解析度控制。

### 8.3 Model size trade-off

RepViT family 結果顯示，模型越大通常帶來更高官方 top-1 accuracy，但 CPU latency 與 miss rate 也快速上升。M0.9 是 CPU 200 ms deadline 下較平衡的選擇；M1.5 和 M2.3 更適合 GPU 或較寬鬆 deadline。1000-class ImageNet subset 也支持此趨勢：M0.6 到 M2.3 的 subset top-1 由 74.4% 提升至 83.7%，但 M1.5/M2.3 的 tail latency 與 deadline miss 風險也較高。

### 8.4 Pipeline overhead

Deadline decomposition 顯示，非模型開銷不可忽略。GPU static M0.9 的平均 inference latency 為 38.047 ms，但平均 non-model overhead 仍有 16.691 ms；model/device switching 時 non-model overhead 更上升到 27.924 ms。這提醒我們 ROS/Python pipeline 的整體 latency 不只由模型 forward pass 決定，也包含 image decoding、resize、message passing、logging、model cache lookup、CUDA synchronization、CPU scheduling 與 container overhead。

### 8.5 QoS、queue policy 與 freshness

Stress 實驗顯示，可靠傳輸與 FIFO queue 雖可避免丟 frame，但可能造成嚴重 freshness delay；在 CPU stress 下，reliable FIFO 的 P95 freshness 達 3098.987 ms。相較之下，best-effort depth=1 搭配 deadline-drop 將 P95 freshness 降至 594.546 ms，但代價是少量 drop 與 3.2% miss rate。修改後 runtime 驗證進一步指出，在無額外 stress 的推薦設定下，即使沒有 drop、defer 或 warm-up 後 miss，平均 freshness 仍為 213.791 ms，P95 freshness 為 317.862 ms，主要受到 publisher image loading、publish overhead 與 callback scheduling 影響。對 real-time perception 而言，最新 frame 往往比完整處理所有舊 frame 更重要，因此 freshness 應和 deadline miss 一起作為評估指標。

### 8.6 CPU affinity 與 admission control

CPU affinity 實驗提供一個重要負面結果：在 CPU stress 下，預載入模型並使用 default scheduling 可以達到 0% miss；但將 inference process 限制在特定核心後，admission controller 判定服務能力不足，導致所有 frames 被 defer。這不表示 CPU affinity 無效，而是表示 affinity 不能只限制 inference node；更合理的實驗設計應隔離 stress process、保留 inference core，或搭配 real-time priority 與 cpuset/cgroup 控制。

### 8.7 Admission control implementation lesson

修改後 runtime 的第一次實測暴露出 admission control 的冷啟動風險：若 `deadline_drop` 在沒有任何 latency sample 前直接用全域預設預測值判定，可能會錯誤地 drop 所有 frames，導致系統永遠沒有機會取得實際 service time。修正後的流程改為「先由 scheduler 根據候選 config 估計 latency，再讓 admission controller 判斷 accept/degrade/drop」，使系統能在 cold-start 後取得樣本並逐步收斂。這個結果說明 admission control 不能只是一個 hard gate，也必須考慮 exploration、warm-up 與 model cache 狀態。

### 8.8 LLM workload 與 vision-language concurrent runtime

LLM 實驗顯示，1B 等級模型不能只用參數量判斷即時可用性。Llama-3.2-1B 在 standalone 200/150 ms per-token deadline 下穩定，但 Gemma-3-1B-it 在 200 ms 已接近或超過邊界。當 LLM 與 RepViT 同時執行時，Gemma 的 miss rate 明顯升高，尤其 mixed stress + 200 ms 下達 100%。這代表 edge AI runtime scheduler 需要針對不同模型建立獨立 latency profile，而不能假設同樣是 1B 模型就有相近 deadline behavior。

LLM 實驗也讓 admission control 的解讀更重要。Gemma 在 80 ms 或 concurrent 100 ms 等嚴格條件下大量 `defer`，使整體 miss rate 看似下降；但排除 defer 後，生成型請求仍全部 miss。因此未來 LLM runtime 報告應同時列出 total miss rate、generated-only miss rate、defer rate 與 output token distribution，否則容易把「拒絕服務」誤解為「成功達標」。

## 9. 結論

本研究完成 EdgeAI-ROS 中 RepViT 自適應推論的完整實驗評估。主要結論如下：

1. **GPU 顯著提升 deadline reliability。** RepViT-M0.9 在 GPU 上於 150/200/250 ms deadline 皆達 0% miss rate，且 decomposition 實驗中平均 inference latency 相較 CPU 約降低 2.22x。
2. **CPU 對 OS pressure 敏感。** CPU mixed stress 下，RepViT-M0.9 的 miss rate 可上升至 29.4%，即使 adaptive policy 已降低解析度仍無法完全避免 miss。
3. **Adaptive policy 需要從解析度擴展到模型與 device。** Resolution-only adaptive 可控制部分負載；true model/device switching 則能在 200 ms deadline 下選擇較高品質模型並維持 0% miss。
4. **RepViT-M0.9 是 CPU 上較好的折衷點。** M0.6 更快但官方 top-1 較低；M1.5/M2.3 在 CPU 上較容易 miss。
5. **較大模型更需要 GPU。** M2.3 在 CPU 上 miss rate 達 74.8%，但 GPU 上平均 inference latency 降至 74.516 ms，miss rate 僅 0.7%。
6. **Freshness 與 non-model overhead 是 real-time pipeline 的關鍵。** QoS depth、queue policy、preprocess、message passing 與 logging 會明顯影響端到端表現，不能只以模型 latency 代表系統 latency。
7. **修改後 runtime 可穩定支援推薦組合。** 在 ImageNet 1000-class subset 上，推薦 model/device switching 設定於 warm-up 後選擇 RepViT-M1.5 GPU，平均 E2E 為 99.096 ms、P95 E2E 為 130.844 ms，200 ms deadline miss rate 為 0%。
8. **Llama-3.2-1B 較適合作為即時 LLM baseline。** Llama standalone 200/150 ms per-token deadline 皆為 0% miss；Gemma-3-1B-it 在 200 ms 已有 miss，且 CPU stress 與 concurrent workload 下明顯惡化。
9. **LLM defer 必須和成功推論分開分析。** Gemma 在嚴格 deadline 下透過 defer 降低總體 miss rate，但 generated-only rows 仍無法達標，因此 defer 是 admission control 的保護行為，不等於模型推論達標。

## 10. 研究限制

本研究仍有以下限制：

1. **ImageNet subset 仍非完整 validation set。** 本研究已由 7 張照片擴展到 1000-class balanced subset，但仍不是完整 50,000 張 ImageNet validation，因此 accuracy 數值應視為 controlled subset estimate。
2. **單一硬體環境。** 結果來自 RTX 3060 Laptop GPU 與目前 Docker/WSL/Windows 環境，不一定可直接外推至 Jetson、Raspberry Pi 或其他 embedded device。
3. **Run-to-run variation。** OS scheduling、Docker resource allocation 與背景程序會造成 latency 波動，因此單次 run 的數值需搭配多組趨勢解讀。
4. **Model/device switching 尚屬初步版本。** 目前使用 heuristic quality score 與 deadline-based selection，尚未整合完整 Pareto frontier、energy model 或 online learning。
5. **Stress workload 較粗略。** CPU stress 與 mixed stress 可模擬負載，但不一定等同真實機器人系統的感測器、控制與網路負載。
6. **CPU affinity 實驗尚未完成最佳隔離設計。** 本次 restricted affinity 造成全數 defer，較適合作為負面案例；未來需以 cpuset/cgroup 分離 stress 與 inference 才能公平評估 affinity。
7. **修改後 runtime 驗證時間仍偏短。** 最新推薦組合 run 使用 `timeout 75s` 控制，得到 87 筆 records；雖足以驗證程式行為與欄位完整性，但仍不足以取代長時間穩定性測試。
8. **LLM rows 數較少。** LLM 實驗每組約 5 至 7 筆 request，足以觀察趨勢，但若要做統計檢定，仍需延長 duration 並重複多次。
9. **Llama 與 Gemma 使用不同 Transformers 版本。** Gemma 3 需要 `transformers==4.50.0` 才能載入，Llama 實驗則使用 `4.45.2`；此差異是模型相容性需求，但仍可能影響 runtime comparison。

## 11. 未來工作

後續可進一步進行：

1. 使用完整 ImageNet validation set 量化 top-1 accuracy、top-5 accuracy 與不同模型/解析度的 accuracy loss。
2. 每組實驗延長至 5 至 10 分鐘，並重複多次以建立 confidence interval。
3. 將 model/device switching 擴展成完整 Pareto scheduler，納入 accuracy、latency、freshness、energy 與 queue state。
4. 在 Jetson 或其他實際 edge device 上重跑，驗證可攜性。
5. 以 tracing 工具進一步分解 deadline miss，包含 preprocess、H2D/D2H transfer、CUDA synchronization、ROS message passing 與 logging。
6. 重新設計 CPU affinity 實驗，使用 cpuset/cgroup 隔離 stress process，並比較 default priority、nice、real-time scheduling 與 pinned inference cores。
7. 評估 QoS depth、best-effort/reliable、deadline-drop 與 frame skipping 對 freshness、accuracy 與控制迴路穩定性的影響。
8. 為 admission control 加入 explicit warm-up/exploration phase，避免 cold-start 時因缺乏 latency sample 而過度 drop。
9. 將 `publish_overhead_ms` 進一步拆分為 image publisher timer jitter、DDS middleware delay 與 subscriber executor waiting time。
10. 將 LLM 實驗延長到 3 至 5 分鐘以上，並針對 generated-only rows、defer rate、TTFT、TPOT 與 output token distribution 建立更完整統計。
11. 評估 LLM 推論最佳化，例如 quantization、ONNX/TensorRT、vLLM 類 serving backend，並重新測試 Gemma 在 300/400/500 ms per-token deadline 下的可行性。
12. 將 vision-language concurrent workload 擴展為多 prompt 類型、多 image rate 與多 executor 配置，分析 ROS 2 executor scheduling 對多模型 workload 的影響。

## 12. 輸出檔案

主要報告與表格：

| 檔案 | 說明 |
|---|---|
| `reports/repvit_adaptive_deadline_report_zh.md` | 初期 adaptive deadline 報告 |
| `reports/repvit_followup_experiments_zh.md` | 後續延長與 stress 實驗報告 |
| `reports/final_cpu_gpu_repvit_research_report_zh.md` | CPU/GPU 最終比較報告 |
| `reports/repvit_model_family_benchmark_zh.md` | RepViT model family benchmark 報告 |
| `reports/imagenet_val_subset_repvit_family_gpu_report_zh.md` | ImageNet 1000-class subset 的 RepViT family GPU accuracy-latency 報告 |
| `reports/edgeai_ros_repvit_complete_research_report_zh.md` | 本完整研究報告 |
| `data/results/final_cpu_gpu_summary.csv` | CPU/GPU policy、deadline、stress summary |
| `data/results/final_cpu_gpu_prediction_summary.csv` | CPU/GPU 分類結果 summary |
| `data/results/repvit_model_family_summary.csv` | RepViT family latency summary |
| `data/results/repvit_model_family_speedup.csv` | CPU/GPU speedup summary |
| `data/results/repvit_model_family_predictions.csv` | RepViT family 分類結果 |
| `data/results/imagenet_val_subset_gpu_repvit_family_summary.csv` | ImageNet subset accuracy-latency summary |
| `data/results/depth_runtime_decomposition_summary.csv` | Deadline decomposition summary |
| `data/results/depth_experiment_overview.csv` | Research-depth experiments overview |
| `data/results/depth_*.csv` | Decomposition、model/device switching、QoS、affinity 等原始實驗資料 |
| `data/results/modified_runtime_recommended_200ms.csv` | 修改後推薦 runtime 組合的原始 frame-level 結果 |
| `data/results/modified_runtime_recommended_200ms_summary.csv` | 修改後推薦 runtime 組合的 decomposition/model-switching/queue summary |
| `data/results/modified_runtime_resolution_only_m09_gpu_200ms.csv` | 同樣 200 ms 設定下 resolution-only GPU M0.9 的原始結果 |
| `data/results/modified_runtime_200ms_comparison_summary.csv` | 同樣 200 ms 設定下 resolution-only 與 model/device switching 的比較 summary |
| `data/results/rerun_resolution_only_m09_gpu_200ms.csv` | 重跑後 resolution-only GPU M0.9 的原始結果 |
| `data/results/rerun_model_device_switching_200ms.csv` | 重跑後 model/device switching 的原始結果 |
| `data/results/rerun_200ms_resolution_vs_switching_summary.csv` | 重跑後 full-run decomposition summary |
| `data/results/rerun_200ms_matched76_comparison_summary.csv` | 重跑後 matched 76-row 公平比較 summary |
| `reports/llm_runtime_experiments_zh.md` | Llama-3.2-1B 與 Gemma-3-1B-it 對齊 LLM runtime 報告 |
| `reports/gemma3_1b_it_runtime_experiment_zh.md` | Gemma-3-1B-it 初步本地 GPU 測試報告 |
| `data/results/llama32_1b_only_summary.csv` | Llama standalone LLM deadline/stress summary |
| `data/results/gemma3_1b_only_summary.csv` | Gemma standalone LLM deadline/stress summary |
| `data/results/llama32_concurrent_llm_summary.csv` | RepViT + Llama concurrent LLM 端 summary |
| `data/results/gemma3_concurrent_llm_summary.csv` | RepViT + Gemma concurrent LLM 端 summary |
| `data/results/llama32_concurrent_vit_summary.csv` | RepViT + Llama concurrent RepViT 端 summary |
| `data/results/gemma3_concurrent_vit_summary.csv` | RepViT + Gemma concurrent RepViT 端 summary |
| `scripts/run_recommended_runtime.sh` | 推薦 runtime 組合的啟動腳本 |
| `tools/summarize_runtime_decomposition.py` | 產生 decomposition 與 freshness summary 的分析工具 |

## 參考文獻

1. Wang, A., Chen, H., Lin, Z., Han, J., & Ding, G. **RepViT: Revisiting Mobile CNN From ViT Perspective.** arXiv:2307.09283 / CVPR 2024. https://arxiv.org/abs/2307.09283
2. THU-MIG. **RepViT official repository.** https://github.com/THU-MIG/RepViT
3. Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. **ImageNet: A Large-Scale Hierarchical Image Database.** CVPR 2009. https://www.image-net.org/
4. Russakovsky, O. et al. **ImageNet Large Scale Visual Recognition Challenge.** International Journal of Computer Vision, 2015. https://arxiv.org/abs/1409.0575
5. Open Robotics. **ROS 2 Humble Hawksbill Documentation.** https://docs.ros.org/en/humble/
6. Docker Inc. **Docker Documentation.** https://docs.docker.com/
7. PyTorch Foundation. **PyTorch Documentation.** https://pytorch.org/docs/stable/
8. NVIDIA. **CUDA Toolkit Documentation.** https://docs.nvidia.com/cuda/
9. Hugging Face. **Transformers Documentation.** https://huggingface.co/docs/transformers/
10. Meta. **Llama 3.2 model family.** https://www.llama.com/
11. Google. **Gemma model family.** https://ai.google.dev/gemma
