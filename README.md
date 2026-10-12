# NLP 2026 Final Project — NewsLens

GitHub: https://github.com/zgyfjch/NLP2026-Final

作业一已单独提交至 https://github.com/zgyfjch/NLP2026-HW1 。本仓库保留共用的分类实验和课程数据，作为大作业的可复现基线。

**学生：2312260 谭易；独立成组；大作业 Track B Demonstration。**

NewsLens 是在本机运行的英文新闻分类与解释系统。作业一提供七组表示方法对照，大作业新增概率校准、人工复核策略、相对类别解释、特征删除实验、输入处理与网页集成。

## 目录

- `HW-1/nyt.csv`、`HW-1/ag.csv`：课程原始数据，不修改。
- `src/`：数据协议、训练、评价、校准、解释与 Demo 服务。
- `templates/`：本地网页。
- `tests/`：10 项功能测试。
- `artifacts/split.json`：数据 SHA-256、种子与全部划分行号。
- `artifacts/results/`：模型结果 JSON、逐测试样本预测 CSV。
- `artifacts/models/`：本地模型；通常不进入源码版本库。
- 作业一实验报告见独立仓库 `NLP2026-HW1`。
- `reports/TeamXX_Final_Report.tex`：大作业 ACL 官方模板论文源码。
- `NLP2026_TeamXX_Demo/`：分阶段项目 PDF。
- `scripts/`：下载、报告生成、检查与打包工具。

## 快速演示

在已经配置好的本机，右键 `Launch_Demo.ps1` 选择用 PowerShell 运行，或执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Launch_Demo.ps1
```

浏览器访问 `http://127.0.0.1:8765`。选择模型、输入英文新闻、点击“分析文本”。演示包含 Binary、Frequency、TF-IDF 三种在线模型；词向量和 BERT 是离线对照，不在页面在线推理路径中。网页明确区分概率、复核提示和相对类别关键词解释。

## 新环境安装

推荐 Python 3.10（本次 3.10.18），至少 8 GB 内存，完整 BERT CPU 训练建议 16 GB。使用 GPU 时选择与显卡兼容的官方 PyTorch wheel；源代码自动使用 CUDA（若可用）。

```powershell
uv venv --python 3.10 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-lock.txt
```

也可用 Python 的 `venv` 与 `pip install -r requirements-lock.txt`。`requirements.txt` 是主要依赖，`requirements-lock.txt` 是本次实际环境的全部精确版本，包含 PDF 生成所需的 `reportlab`、`pypdfium2` 和 `Pillow`。ACL 编译需要 XeLaTeX。中文作业一报告使用 Windows 的微软雅黑字体；其他系统需在生成脚本中配置可用中文字体。训练不依赖这些排版工具。

若数据不随代码包分发，从课程文件夹取得原始 `nyt.csv` 和 `ag.csv` 放入 `HW-1/`。项目打包脚本会包含课程数据以便课程内部复核；数据版权仍归原提供方。不要把其他同学的提交登记表上传到自己的仓库。

## 下载指定模型

```powershell
.venv\Scripts\python.exe scripts\download_assets.py
```

下载 Stanford 官方 `glove.6B.zip`，提取 `glove.6B.100d.txt`；下载 `google-bert/bert-base-uncased` 的 tokenizer、config 与 safetensors。`cache/bert-base-uncased/provenance.json` 保存 revision。脚本支持续传 `.part` 文件，不在源码中保存代理地址、访问令牌或账号。

网络需要代理时，在当前终端设置自己的 `HTTPS_PROXY`、`HTTP_PROXY` 再运行，不要把认证信息写入仓库。大权重约 1.3 GB 下载量，解压、BERT 检查点及训练还需额外磁盘空间。

## 完整运行

```powershell
$env:PYTHONHASHSEED='42'
.venv\Scripts\python.exe src\train_classical.py --methods binary frequency tfidf glove w2v_ag w2v_nyt
.venv\Scripts\python.exe src\train_bert.py --model cache/bert-base-uncased --batch-size 16 --threads 6
.venv\Scripts\python.exe src\project_analysis.py
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe src\demo.py
```

`Run_Experiments.ps1` 可按顺序执行上述下载与实验。仅从 BERT 已完成 epoch 检查点续训：

```powershell
.venv\Scripts\python.exe src\train_bert.py --model cache/bert-base-uncased --resume
```

不会自动恢复半轮中的 batch；`resume.pt` 记录的是最近完整 epoch。训练三轮，最后使用验证 Macro-F1 最佳的模型做测试。`artifacts/bert_progress.json` 提供实时进度，`bert_history.json` 提供已结束轮次结果。CPU 训练较长，运行期间不要关闭训练进程或让电脑休眠。

## 实验协议

先按 seed=42 随机打乱 NYT，再按 80/10/10 切分。训练 9,215，验证 1,152，测试 1,152。所有方法共享同一组行号；程序验证数据哈希，拒绝静默复用修改后数据的划分。

1. Binary、Frequency、TF-IDF 只在训练集构造词表与 IDF，分类器均为 Logistic Regression。
2. GloVe 为 100 维 `glove.6B`；Word2Vec 分别在 AG 全部文本与 NYT 训练文本训练，均 100 维。只平均已登录词，零有效词文档用零向量。缩放器只拟合训练集。
3. 每个线性模型从 C=0.1/1/10 按验证 Macro-F1 选择，不用测试集调参。
4. BERT 指定模型、max_length=64、3 epochs、全参数微调，未用小模型替代。
5. 大作业的 TF-IDF 概率温度只用 576 条验证文本拟合，阈值只用另外 576 条选择。测试集仅做报告。
6. 原始要求说三种 BoW 但只列两种；本实现补充 TF-IDF。原始文件未给 seed 数值，选用 42 并公开记录。

类别分布偏斜，72 条文本重复，8 条测试文本在训练中出现。主结果保留课程划分，并另报去掉这 8 条后的 TF-IDF 结果。该检查不等于事件级去重。全文词袋和 BERT 的 64 WordPiece 输入不等长，报告明确限制模型比较结论。

## 核心解释

线性模型 `z_c = b_c + Σ w_cj x_j`。对于首选类别 a 与次选 b，词 j 的贡献为 `x_j × (w_aj - w_bj)`。全部词贡献之和加截距差应等于 `z_a-z_b`，测试检查该等式。展示只列正贡献最大的词，不是对因果关系或新闻真假的判断。

删除实验在前 200 条固定测试样本中，分别将前五个正贡献特征或同数量随机特征置零，不重新归一化。它是特征空间诊断，不等同于删除文本后重新分词。

## 报告生成

```powershell
python scripts/build_pdfs.py
python scripts/build_acl.py
xelatex -interaction=nonstopmode -halt-on-error -output-directory=reports reports/TeamXX_Final_Report.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=reports reports/TeamXX_Final_Report.tex
python scripts/render_pdfs.py
```

生成 PDF 后逐页检查。`reports/acl.sty` 直接来自 [ACL 官方仓库](https://github.com/acl-org/acl-style-files)，未改动页面参数。最终正文目标 8 页，参考文献另页。团队编号尚未由课程提供，暂用 TeamXX；不能将学号自行当作课程分配的团队编号。

## 来源与贡献

- [Word2Vec 原论文](https://arxiv.org/abs/1301.3781)
- [GloVe 原论文](https://aclanthology.org/D14-1162/) 与 [官方向量](https://nlp.stanford.edu/projects/glove/)
- [BERT 原论文](https://aclanthology.org/N19-1423/) 与 [指定模型](https://huggingface.co/google-bert/bert-base-uncased)
- [温度缩放与校准](https://proceedings.mlr.press/v70/guo17a.html)

谭易独立成组，负责项目全部角色。使用 Codex 辅助需求分析、代码和文稿起草、运行与调试、报告制作；算法库和预训练模型来自上述第三方。性能来自实际运行，未编造结果。提交前应亲自审阅代码、运行 Demo，并能解释核心流程。贡献声明提供详细区分。
