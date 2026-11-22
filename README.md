# Judgement Bias Studio (SEEG + Behavior + Clinical + Localization SOP)

面向 **Auditory Judgement Bias** SEEG 课题的标准化、傻瓜式前后端分析工作台，支持多被试管理与插件化扩展。

## 核心功能模块

1. **Step 1 · 数据体检与并口触发对齐**
   - 自动扫描术前 MRI / 术后 CT 薄层序列、解析日本光电 EDF 电极杆与缺失跳号（自动兼容中文 GB18030 标注）。
   - 解码 `POL DC02–DC07` 6-bit 并口脉冲（声音呈现 + 左右键按键），并内置**墙钟线性时钟拟合漏标自动补齐**（修正晶振漂移，毫秒级精度）。
   - 支持可选导入医生术后核对的电极位置表（`.xlsx/.csv`）。
2. **Step 2 · 临床数据录入（拍照 / 上传 + AI 自动填表）**
   - 支持多文件/照片上传、摄像头直接拍照，以及 OpenCode Go（默认 `deepseek-v4.1-flash-1017`）视觉+文本一键提取 17 项结构化临床协变量，并自动并入跨被试汇总表。
3. **Step 3 · Judgement Bias 行为学分析**
   - 自适应效价对齐（绝不写死高低音/左右键），页面内嵌 4 张交互式 SVG 图表（心理测量曲线、RT ± SD、分 Block 稳定性、7 点 VAS 情绪轨迹）并支持高清 PNG 与 CSV 导出。
4. **Step 4 · 电极定位与 3D 脑核对**
   - 复用同级 `seeg-workbench` 的检测/刚体配准/WebGL2 3D 脑渲染内核，支持选序列、调阈值、切面剖切、MNI 脑区与医生位置表双重标注。
   - ⚠️ **本步骤依赖同级的 `seeg-workbench` 仓库**（`electrode_pipeline.py` 会在导入时加载它的 `electrode_detect` / `pipeline` / `volume_preview`）。若该目录缺失，整个后端**无法启动**，详见「依赖与启动」。
5. **Step 5 · SEEG 标准化预处理（PNAS 2025 SOP）**
   - 对齐 Zhang et al., *PNAS* 2025：CAR（或 Bipolar）重参考 → 250 Hz（或 1000 Hz）降采样 → 50 Hz 180 阶 Hamming FIR 陷波 → 0.3 Hz 2 阶 Butterworth 高通 → `Q3 + 2.3×IQR` 癫痫样试次剔除 → 导出含完整行为 metadata 的标准 MNE `-epo.fif`。
6. **Step 6 · 可插拔扩展分析插件**
   - 在 `backend/app/plugins/` 下新增继承 `AnalysisPlugin` 的 `.py` 文件即可自动注册到前端 UI。
   - 内置插件：
     | 插件 | 作用 |
     |---|---|
     | `condition_stats` | **条件效应 × 手侧混淆诊断（A/B/C 三口径 + 家族校正）** — 见下方「手侧混淆」 |
     | `erp_preview` | 锁相诱发 GFP 描述性预览（非效应检验） |
     | `band_power_profile` | 各电极杆频段功率热图 |

## 依赖与启动

```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8780
```

macOS 上双击 `启动 JudgementBias 工作台.command`（默认 `http://127.0.0.1:8780`）。

**Step 4 的目录约定**：`backend/app/electrode_pipeline.py` 在 **模块导入期** 就需要
`<本仓库父目录>/seeg-workbench/backend/app/{electrode_detect,pipeline,volume_preview}.py`。
这个仓库没有公开它，所以：

- 如果只需要 Step 1–3 / 5–6，**不要 import Step 4**，或把 `seeg-workbench` 放到同级目录；
- 缺失时的报错信息会明确指出期望路径，但它发生在 import 阶段，
  表现为 **整个 app 启动失败** 而不是「Step 4 不可用」。

> 注意：所有被试数据、临床上传照片及本机 AI API Key 均存储在 `workspace_data/` 中，已通过 `.gitignore` 严格排除，不会上传至 Git 仓库。

## 分析口径（Step 6 插件共用）

统计约定集中实现在 `backend/app/stats_core.py`，**不要在各插件里另立口径**：

| 项 | 约定 |
|---|---|
| 置换 p | 一律 `(k+1)/(n+1)`，不用 `k/n`（否则 1000 次置换的最小 p 报成 0） |
| 置换方式 | **block 内**打乱；block 携带疲劳/适应/累积奖励等真实结构，全局打乱会虚高零分布 |
| 家族校正 | `max|t|` over（通道 × 窗口）**在同一口径内**；p 属于家族，不属于事后挑的峰通道 |
| 主窗口 | 刺激后 **0–200 ms**（早于任何可测的 premotor 准备） |
| 次窗口 | `150–350 ms` 等必须注明「可能含 premotor / 按键准备」，不得单独作为唯一证据 |
| 重基线 | 分析阶段统一改 **刺激前 −200…0 ms**；预处理的「整段均值」会把刺激后电位摊回刺激前，产生假 t |

## 手侧混淆（必读，规则 R1–R8）

### 为什么必须处理

范式里每个锚点对应**固定的正确按键**（001-10.30：高音=左、低音=右；002-11.06：高音=右、低音=左），
所以「高 vs 低」「正 vs 负」与「左手 vs 右手」在锚点试次里**几乎是同一个变量**。2026-11-12 实测：

| 受试者 | 锚点里手的符合率 | r(pitch, hand) 全体 | r(pitch, hand) 中音 | VIF(pitch) | 条件数 |
|---|---|---|---|---|---|
| 001-10.30 | 124/126 | −0.838 | −0.297 | 3.37 | 26.0 |
| 002-11.06 | 126/126 | +0.851 | +0.289 | 3.67 | 24.3 |

结论：**锚点内部不可能把手"校正"掉**；加 `hand` 协变量会把条件效应削掉一半以上
（001 CIL8：t 5.38 → 2.07；002 CIR7：4.66 → 3.03），但这不是"去掉了手"，
而是两个共线变量在分方差。

### 规则

**R1 先报一致性，再谈解释。** 任何以锚点/效价为条件的对比，必须先算
`hand_agrees_with_mapping`（手侧与映射预测一致的比例）。≥95% 时，该对比在表头与正文里
一律写「条件（与手侧共变）」，**不得**写成「效价效应」「决策效应」「判断偏倚的神经相关」。

**R2 三口径并列，不许只挑一个。** 每个条件效应至少给出：

| 口径 | 设计 | 作用 |
|---|---|---|
| **A** | 锚点 only，`y ~ pitch_lin` | 与旧结果可比，但含手 |
| **B** | 全部 main，`y ~ pitch_lin + hand + block` | 手进入模型；受共线性影响，只能当下界/参考 |
| **C** | 中音 only，`y ~ pitch_lin + hand` | 手有变异时的"较干净"估计，但试次少、功效低 |

报告里三个 t 值必须同时出现（家族校正后的 max|t| 与探针通道的单通道 t 都要）。
**只报 A 或只报 B 视为选择性报告。**

**R3 共线性门槛。** 同时报 `VIF(pitch)`、设计矩阵条件数、`r(pitch, hand)`。
判据：`VIF > 5` 或 `|r(pitch,hand)| > 0.7` → 标为「**不可分离**」，
结论只能写「与手侧不可分」，不得给出方向性解释。

**R4 手图诊断（决定一个 ROI/通道能不能解释）。** 用中音试次估计「手效应图」`t_hand`
（要求少数手 ≥ 20 个试次，否则写「中音手侧变异不足，无法诊断」），然后报：

- `r(条件图, 手图)`（同窗口、同通道集）
- top-k 通道的重合率（**必须相对机会水平 `k/通道数` 看**：24 通道时 top-10 的随机重合就有 42%）
- 峰通道是否落在手图上 `|t| > 2` 的通道里

判据：`|r| > 0.5` 或（重合 ≥ 50% **且** ≥ 3× 机会水平）→ 该区域的条件效应标为
「**运动混淆候选**」，不进结论段；`|r| < 0.3` 且峰通道不在手图上 → 可写「不能用按键解释」，但仍注明共线。

**R5 窗口约束。** 主结果用**刺激后早期窗口 0–200 ms**（早于任何可测的 premotor 准备）；
`150–350 ms` 必须注明「可能含 premotor / 按键准备」，且不能单独作为唯一证据。
不要在 RT 分布的中位数之后取窗（001 锚点 RT ≈ 1099 ms，002 ≈ 669 ms）。

**R6 RT / EMG 校验。** 逐通道报 `r(单试次幅度, RT)`，并报 `|r| > 0.3` 的通道数占比与峰通道数值；
有 EMG 的会话（001 EMG1–2、002 EMG1–4）同样报 `r(幅度, EMG)`。
EMG 相关性需要原始连续数据，插件内算不了，须在预处理阶段单独报告。

**R7 不做的事。**

- 不剔除"错误按键"的锚点试次来制造手平衡（001 只有 2 个，剔了也不会平衡，且会引入选择偏倚）。
- 不按 RT / 反应时分组"配平"手侧。
- 不用事后挑通道、挑窗口来规避手图重叠。
- 不把 `hand` 与 `pitch` 同时放进只含锚点的模型（完全共线，结果不可解释）。
- 不在报告里把「条件（与手侧共变）」简写成效价/情绪/决策。

**R8 设计层面的优先顺序。** 若要真正去掉手，优先级：
(1) 采集时让同一锚点**双手都按键**（counterbalance）→ `hand` 成为真正的因子；
(2) 分析上只用中音试次（手与音高部分解耦，但要 ≥20 个少数手试次）；
(3) 用反应前早窗（0–200 ms）；
(4) 最后才是协变量校正，且必须与 A/C 并列。
没有 (1) 的数据，**任何写法都不能声称"已控制手"**，只能是"报告了手的影响范围"。

### 怎么跑

Step 6 页面运行 `condition_stats` 插件（或直接调 API）：

```bash
curl -X POST http://127.0.0.1:8780/api/subjects/<subject_id>/plugins/condition_stats/run \
  -H 'Content-Type: application/json' \
  -d '{"params": {"n_perm": 1000, "baseline": "pre_stim", "top_k": 10}}'
```

产物（在 `workspace_data/subjects/<subject_id>/plugins/condition_stats/`）：

| 文件 | 内容 |
|---|---|
| `CONDITION_STATS.md` | 人读报告：R1 设计事实、R3 共线性、R2 三口径、R4 手图、R6 RT、可以写/不可以写 |
| `s5_hand_diagnostics.csv` | 手侧符合率、对比标签、`can_claim_hand_controlled` |
| `s5_hand_models.csv` | 每个口径一行：n、max&#124;t&#124;、家族 p、峰通道/窗口、VIF、条件数、混淆判定 |
| `s5_hand_channels.csv` | 逐通道：A/B/C 的 t、手图 t、`r(RT)` |
| `condition_stats.png` | 条件图 × 手图散点 + 三口径 max&#124;t&#124; 柱状图 |

插件内部拒绝：「只报一个口径」「手图不可用却下结论」「没有 counterbalance 却说已控制手」。
`test_condition_stats.py` 把这些口径固化成断言（零假设标定、FWL 与直接 OLS 等价、
`(k+1)/(n+1)` 下限、block 内置换、top-k 相对机会水平等）。

## 复现（本地 002 项目）

```bash
cd "/mnt/NTFS2T/BaiduNetdiskDownload/002 WYF/code/seeg-002-analysis"
PY=~/.local/opt/freesurfer/python/bin/python3
$PY -I scripts/40_hand_diagnostic.py      # → outputs/HAND_DIAGNOSTIC.md
```

产物：`outputs/HAND_DIAGNOSTIC.md`、`outputs/tables/s5_hand_{diagnostics,models}.csv`。

## 测试

```bash
python -m pytest tests/ -q     # 需要 numpy/scipy/pandas/matplotlib/fastapi/openpyxl/Pillow
```

> `tests/test_studio_core.py` 里 `from backend.app import main` 会触发 Step 4 的
> `seeg-workbench` 导入。在没有该同级仓库的机器上，需要先按「依赖与启动」处理，
> 否则测试收集阶段即失败。新增的 `tests/test_condition_stats.py` 只依赖
> `stats_core` + `plugins.condition_stats`，不受此影响。
