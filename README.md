# 心脏离子通道预测与分子优化系统

本项目面向药物心脏安全性评价，提供 hERG、Cav1.2 和 Nav1.5 三个离子通道的独立 XGBoost 基线模型，以及“预测—改造—复审”Agent 闭环。

三个通道均包含分类模型和回归模型，并采用相同的输入特征：

- Morgan 指纹：半径 2、2048 位
- MACCS 指纹：167 位
- RDKit 理化性质描述符：18 项

模型采用独立目录持久化，后续可以单独替换某个通道的模型，而不需要修改 Agent 的数据接口。

## 数据来源

- **hERG**：ChEMBL 37，靶点 `CHEMBL240`。保留单位为 nM 的精确 IC50 测量值，按 InChIKey 去重，并对同一化合物的重复测量取中位数，共 9,628 个化合物。
- **Cav1.2**：Cardiosim-Tox 开发集（Zenodo 19902546），共 1,139 条数据。
- **Nav1.5**：Cardiosim-Tox 开发集（Zenodo 19902546），共 4,585 条数据。

整理后的数据位于 `data/dev/`，特征文件位于 `data/features/`，持久化模型位于 `artifacts/models/`。

## 基线模型

当前版本使用 XGBoost 分别训练三个通道的分类与回归模型。分类任务预测是否为通道阻滞剂，回归任务预测 pIC50，并换算得到 IC50（μM）。

训练脚本先进行一次固定随机种子下的 80/20 留出集基线评估，再使用全部数据重新拟合并持久化最终模型。这里的留出集指标用于验证流程是否跑通，不作为严格的外部验证结论。

当前基线结果：

| 通道 | 样本量 | 分类 ROC-AUC | 回归 R² |
| --- | ---: | ---: | ---: |
| hERG | 9,628 | 0.828 | 0.564 |
| Cav1.2 | 1,139 | 0.954 | 0.902 |
| Nav1.5 | 4,585 | 0.953 | 0.880 |

## 训练模型

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe scripts\download_herg_chembl.py
.\.venv\Scripts\python.exe scripts\build_features.py
.\.venv\Scripts\python.exe scripts\train_final_models.py
```

其中：

- `download_herg_chembl.py`：从 ChEMBL 下载并整理 hERG 数据。
- `build_features.py`：为三个通道生成 Morgan、MACCS 和 RDKit 描述符特征。
- `train_final_models.py`：完成留出集评估，并在全量数据上训练最终分类和回归模型。

## 启动 Agent 系统

```powershell
.\.venv\Scripts\python.exe run.py
```

程序会自动选择从 `8010` 开始的可用端口。默认访问地址为：

```text
http://127.0.0.1:8010
```

## Agent 工作流程

系统由四个环节组成：

1. **初始毒性评估**：对输入 SMILES 同时输出 hERG、Cav1.2 和 Nav1.5 的分类概率、pIC50 与 IC50。
2. **分子结构优化**：由大语言模型根据当前预测结果、检索证据和上一轮失败原因生成候选结构。
3. **候选毒性复测**：使用相同的三个本地模型重新预测候选分子。
4. **多目标复审**：综合判断 TdP 风险、通道约束、类药性、结构相似性与合成可行性；不满足要求时最多继续优化三轮。

当前 Agent 使用 `gpt-5.6-sol` 生成候选结构。服务地址、模型名称和密钥通过 `.env` 配置，`.env` 已加入 `.gitignore`，不会被纳入版本管理。

## MICE Model 5 风险分级

TdP 风险使用 hERG 与 Cav1.2 的预测效力差分：

```text
CavD = log10(IC50_Cav1.2 / IC50_hERG)
```

该方向与 CiPA 参照药的预期结果一致：dofetilide 为高风险，verapamil 为低风险。接口同时返回反向表达式 `log10(IC50_hERG / IC50_Cav1.2)`，用于避免论文描述和程序实现之间出现符号歧义。

Nav1.5 不参与当前 MICE Model 5 风险公式，而是作为分子改造阶段的硬约束：候选分子的 hERG IC50 必须提升，同时 Nav1.5 IC50 不得降低，以减少传导减慢和 QRS 延长风险。

## 多目标复审条件

候选分子只有同时满足以下条件才会被判定为优化成功：

- MICE TdP 风险降至 `low`
- hERG IC50 相比原始分子提升
- Nav1.5 IC50 不低于原始分子
- SA Score 小于 7
- QED 不低于 0.40
- Lipinski 违例数不超过 1
- 与原始分子的 Morgan-Tanimoto 相似度不低于 0.40

目前尚未接入具体药效靶点模型，因此结构相似度仅作为“活性保持代理”，不能表述为真实的药理活性预测。

## CiPA 参照药验证

以下接口执行参照药验证：

```http
GET /api/reference-validation
```

当前要求：

- Dofetilide：预测为高风险
- Verapamil：预测为低风险

只有两个参照药均符合预期时，接口中的 `passed` 才会返回 `true`。

## 当前暂不实现

按照现阶段课题计划，以下功能暂不包含在本版本中：

- 三通道多任务学习
- SHAP 结构警示定位
- 应用域判断与域外拒绝

当前三个独立基线模型用于先跑通完整 Agent 系统。后续完成多任务模型后，只要保持三通道预测的输出字段不变，即可替换现有模型而无需重写下游流程。

## 测试

运行后端测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

构建前端：

```powershell
cd frontend
npm run build
```
