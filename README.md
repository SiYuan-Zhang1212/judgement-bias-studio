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
5. **Step 5 · SEEG 标准化预处理（PNAS 2025 SOP）**
   - 对齐 Zhang et al., *PNAS* 2025：CAR（或 Bipolar）重参考 → 250 Hz（或 1000 Hz）降采样 → 50 Hz 180 阶 Hamming FIR 陷波 → 0.3 Hz 2 阶 Butterworth 高通 → `Q3 + 2.3×IQR` 癫痫样试次剔除 → 导出含完整行为 metadata 的标准 MNE `-epo.fif`。
6. **Step 6 · 可插拔扩展分析插件**
   - 在 `backend/app/plugins/` 下新增继承 `AnalysisPlugin` 的 `.py` 文件即可自动注册到前端 UI。

## 启动方式

双击 `启动 JudgementBias 工作台.command`（默认监听 `http://127.0.0.1:8780`），或在终端运行：

```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8780
```

> 注意：所有被试数据、临床上传照片及本机 AI API Key 均存储在 `workspace_data/` 中，已通过 `.gitignore` 严格排除，不会上传至 Git 仓库。
