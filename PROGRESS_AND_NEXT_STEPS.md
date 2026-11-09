# 个人实验进度与接手说明

更新日期：2026 年 11 月 9 日。本文记录**实际完成**与**仍待完成**的工作，供本人或之后的协作者接续使用。实验目标是用冻结的 CNNDetection 与 UniversalFakeDetect（CLIP ViT-L/14）检测器，研究局部 AI 编辑面积比 `r` 与前景/背景位置 `L` 对图像级检测的影响。这里的独立工作是诊断评估；组内数据集建设、来源链输出、可选定位实验和模型微调不算作这项核心评估的已完成结果。

## 一、当前可确认的状态

| 项目 | 已完成或未完成 | 证据 / 说明 |
|---|---|---|
| 原图清点 | 已完成 | 两个 ZIP 共 300 张可解码照片：iPhone 14 Pro 100 张、iPhone 16 Pro 200 张；见 `data/root_manifest.csv` 与 `data/inventory_summary.json`。原 ZIP 未修改。 |
| 重复与场景检查 | 部分完成 | 没有完全相同的 SHA-256 文件，也没有 pHash 汉明距离 ≤6 的照片对；人工缩略图检查另标出 25 对相关场景候选，合并成 15 个多图候选组。候选关系尚需拍摄者逐一确认；见 `data/visual_scene_candidates.csv` 和 `data/contact_sheets/`。 |
| 开发/验证/测试划分 | **仅暂定** | `data/provisional_root_manifest.csv` 为 180/60/60；目前审计没有发现候选组跨划分或相同哈希跨划分，见 `data/provisional_audit.json`。场景关系未确认，所以绝不能称其为冻结测试集。 |
| 开发图准备 | 已完成 | 选出 20 张暂定开发集原图，放在 `pilot_originals/`；另制成最长边 1600 像素的正向 PNG，在 `pilot_working_png/`，尺寸与哈希见其中的清单。这 20 张按方便性选取，不能代表全部 300 张。 |
| 前景标注 | **未完成** | `pilot_working_png/foreground_review.csv` 尚待逐图审定。试过两个矩形初始化的分割候选，因溢到背景而舍弃，未保留为有效前景标签。 `mask_editor.html` 可手绘候选，但标注者必须先看原图确定前景，再看编辑 mask。 |
| 局部 AI 编辑 | **未获得合格样本** | 本地 SD v1.5 inpainting 做了 5 次开发试验：1 次输出失败，4 次有输出图但目标物/标记没有清晰出现；全部拒绝。原始输入、输出、参数和逐次判定见 `pilot_trial/QUALIFICATION.md`。目前没有可计入正式数据集的编辑图。 |
| 冻结检测器接入 | 已完成开发集推理检查 | 官方 CNNDetection 与 UniversalFakeDetect 仓库和权重已在 `third_party/`、`model_cache/`；代码版本、SHA-256、输入变换和分数方向见 `DETECTORS.md`。两套检测器都在同一批 20 张**未编辑**开发图上跑通，共 40 条记录，见 `data/pilot_pristine_scores.csv`。这些分数只证明推理流程可运行，不能表明局部编辑检测性能。 |
| 测量与分析代码 | 已完成初版 | `fyp_eval.py` 包含清点、分组、划分审计、mask 面积和位置测量；`analyze.py` 包含验证集阈值、分层 AUROC、按原图聚类的不确定性和配对位置比较。8 项合成数据测试通过，见 `test_results.txt`；测试不是实测性能。 |
| Interim Report I | 已有文件 | 上级目录已有 `Interim_Report_I_GUO_Hoi_I.pdf`、`.docx` 和 `Interim_Report_I_content.md`。本轮数据与模型试验没有自动改写该报告。引用任何本轮进展前，应按报告的提交时点和证据重新核对措辞。 |

## 二、研究规则：后续工作必须保持

1. **个人核心问题**：编辑面积比 `r = |M|/(H×W)` 与位置 `L`。`M` 是实际交给编辑器的操作 mask，不得用编辑前后像素差图替代。面积分组和位置阈值在 `PROTOCOL.md` 中；结果只能在足够样本和分组成立时解释。
2. **先定义前景**：在未编辑原图上独立标注前景 `F`；不要拿编辑 mask 的边框或点去提示 SAM。完成并保存 `F` 后才计算 `|M∩F|/|M|`。含糊场景标为 uncertain，不强迫二分。
3. **原图层级隔离**：同一原图及其全部派生图必须留在同一 development、validation 或 test 划分；重复场景/连拍也要保守地合组。确认场景候选后重新生成并冻结清单，保存哈希。不要利用测试分数改划分。
4. **检测器预处理分开**：CNNDetection 使用其官方 `demo.py` 默认无裁剪输入路径；CLIP 检测器使用其官方 `validate.py` 的 `CenterCrop(224)` 与 CLIP 归一化。二者共同的 1600 像素工作图转换须另外记录。不得为了让分数好看而调整其中一路输入。
5. **评估与阈值**：每个模型仅在未编辑验证集原图上确定 5% FPR 阈值，再固定用于测试集。只有合格编辑样本、对齐 mask、冻结划分与真正的测试预测都具备时，才计算面积和位置结论。

## 三、下一次接手的具体顺序

### Gate 1：确认场景关系并冻结划分

1. 打开 `data/contact_sheets/contact_01.jpg` 至 `_12.jpg`，结合 `data/visual_scene_candidates.csv` 核对候选照片对；最好由拍摄者确认是否属于同一场景、连拍或近似机位。记录每对的决定和简短理由。
2. 依据确认结果形成最终场景组。所有同根及应合并的同场景图置于同一划分。保留已经用于模型与编辑器开发试验的根图在 development，或从最终测试中排除。
3. 运行 `fyp_eval.py audit` 检查跨划分重复/组冲突，随后冻结最终清单并记录 SHA-256。当前 `provisional_root_manifest.csv` 只是参考，不能直接称为最终划分。

### Gate 2：先在原图上审定前景

1. 从 20 张开发图中选择有明确主体且同时有可编辑背景的根图。先只看 `pilot_working_png/<root_id>.png`，记录主体定义、遮挡与歧义。
2. 可用 `mask_editor.html` 手绘前景；一个根图保存一个与工作图同尺寸的 8 位 PNG。审查是否漏标/溢出。无法稳定定义主体的图标 uncertain。
3. 在前景 mask 审定后，才设计前景和背景操作 mask。不要让预计编辑位置反向决定主体。

### Gate 3：换用或改进可导出 mask 的编辑流程，先过资格试验

1. 目前本地 SD v1.5 配置连续 5 次没有生成可辨认目标，**不要批量运行它**。选择能明确保存输入操作 mask 的局部 AI 编辑器/模型，先只在 development 图上试 1–2 张。编辑器选择尚未确定。
2. 为每张试验记录原图 ID、编辑图、输入操作 mask、提示词、模型/工具版本、随机种子、步骤、裁剪/补丁选项、输出格式和时间。建议同时保存未编辑但经过相同导出路径的控制图。
3. 在查看检测器分数之前，人工检查目标是否清晰、图像是否自然、输出与 mask 是否同尺寸、mask 外像素是否变化、边界是否异常。失败样本保留在试验记录中，但不得进入正式 `edited_samples.csv`。
4. 合格后，用 `fyp_eval.py measure-samples` 确认面积与位置。先做同根、面积相近的前景和背景编辑，再扩展到预设面积档；记录做不到的档位，不能编造覆盖。

### Gate 4：冻结模型与正式分析

1. `DETECTORS.md` 已有官方版本与权重哈希。把最终接受的源图、编辑图和导出控制图都交给两套冻结检测器，记录每条分数、root ID、split、面积档、位置标签和处理路径。
2. 在验证集的未编辑控制图上求每模型固定阈值。然后在测试集报告可用样本数、独立根图数、AUROC、固定阈值召回率、按根图聚类的不确定性、面积趋势和面积匹配的前景/背景配对比较。排除原因必须可追踪。
3. 如合格样本仍太少，报告方法和试验失败的事实，不对面积或位置效应作统计结论；及时与导师讨论数据和时间风险。

## 四、快速定位与复核命令

在 `individual_evaluation/` 目录执行。命令不会自动完成尚未有人审核的场景和前景决策。

```bash
.venv/bin/python -m unittest discover -s tests -v
python3 fyp_eval.py audit data/provisional_root_manifest.csv
USE_TF=0 USE_FLAX=0 .venv/bin/python run_frozen_detector_pilot.py --detector cnn --device mps
USE_TF=0 USE_FLAX=0 .venv/bin/python run_frozen_detector_pilot.py --detector clip --device mps
```

冻结检测器复跑会覆盖同名开发集分数记录；只有需要检查推理时才复跑。`run_inpaint_trial.py --help` 给出单次编辑试验参数。正式样本的 CSV 模板是 `data/edited_samples_template.csv`；正式预测的模板是 `data/predictions_template.csv`。不要把 `pilot_trial/` 里的失败输出复制进去。

## 五、暂停点

本次执行停在**完成第五次编辑器资格试验、记录其失败，并写好本文件之后**。没有后台生成任务需要继续等待。下一步首先要决定合格的 mask 导出编辑流程，并核定场景候选；在这两项完成前，不应声称已有冻结测试集、合格局部编辑数据或检测器对小面积/背景编辑的性能结论。
