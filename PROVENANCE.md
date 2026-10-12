# 数据与第三方资源来源

课程原始文件：`作业一要求.docx`、`自然语言处理课程大作业要求.docx`、`HW-1/nyt.csv`、`HW-1/ag.csv`。原始文件保持不变。提交登记 CSV 含其他同学信息，不放入交付包。

ACL 模板：<https://github.com/acl-org/acl-style-files>，本次取得 commit `d5adc823ff0f80f98c80405ca0ab66c68e684409`。`reports/acl.sty` 和 `reports/acl_natbib.bst` 为原版副本，未更改。

BERT：<https://huggingface.co/google-bert/bert-base-uncased>。实际 revision 见 `cache/bert-base-uncased/provenance.json`，打包时另存入 `artifacts/model_provenance.json`。下载官方 `model.safetensors`，训练程序新建三分类头并微调全部参数。

GloVe：<https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip>。选用 `glove.6B.100d.txt`，并非其他维度或其他语料版本。使用 Python zipfile 解压时验证 CRC；最终资源清单另记 SHA-256。

Word2Vec：Gensim 4.4.0 从课程提供的 AG 文本或 NYT 训练集自行训练；算法思想引用 Mikolov 等 2013 年论文。

依赖安装来自 PyPI / 清华 PyPI 镜像，准确包版本在 `requirements-lock.txt`。库代码不计为项目自身创造；项目代码、数据处理协议、系统集成与验证范围见贡献声明。

检索并核对的文献来源：

- Devlin et al. 2019: <https://aclanthology.org/N19-1423/>
- Pennington et al. 2014: <https://aclanthology.org/D14-1162/>
- Mikolov et al. 2013: <https://arxiv.org/abs/1301.3781>
- Guo et al. 2017: <https://proceedings.mlr.press/v70/guo17a.html>

工具辅助：Codex 参与实现与文档制作。使用范围公开披露，不将模型输出冒充真实实验或已经发生的人类审阅。

公开课程/项目实例仅作为提交组织与可复现文档的参考：

- [NKU 课程作业示例](https://github.com/Burhan-Hu/NLP-HW-1)：参考其 README 对环境、数据、实验和输出的分节说明；没有复制其作业代码、预测或成绩。该示例使用相同课程数据，不构成本项目独立实验结果。
- [公开新闻标注项目](https://github.com/asaee/ML_capstone_springboard)：参考其将数据检查、模型、部署与测试分别叙述的报告结构；没有复用其数据或部署代码。本项目选择本地 Flask、验证集校准、分数可加解释与删除诊断作为自己的系统内容。
- [Stanford CS230 英文新闻分类报告](https://cs230.stanford.edu/projects_winter_2019/reports/15795079.pdf)：参考学术报告按问题、方法、实验、分析组织的形式；不使用其模型实现或实验数值。

这些实例不是本项目方法和结果的学术依据；方法引用仍以原论文为准。
