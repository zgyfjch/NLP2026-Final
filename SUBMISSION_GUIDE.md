# 提交指南

学生：2312260 谭易。独立成组，Track B，主题 NewsLens 可解释新闻分类系统。

大作业 GitHub 仓库：https://github.com/zgyfjch/NLP2026-Final

作业一 GitHub 仓库：https://github.com/zgyfjch/NLP2026-HW1

## 作业一

课程要求：实验报告 PDF、完整代码、运行说明，通过包含所有文件的 GitHub 链接提交。用户确认更新后的作业一和大作业阶段提交截止时间均为 **2026 年 10 月 12 日 18:00（北京时间）**；以在线表格显示为准。

报告：`reports/HW1_2312260_TanYi.pdf`。

代码：`src/`、`scripts/`、`tests/`、`templates/`；环境：两个 requirements 文件；说明：`README.md`。原始课程数据保留在 `HW-1/`。

提交前必须确认 `artifacts/verification.json` 的 `all_complete` 为 `true`，其中七个实验均为 `verified`。该检查会从逐样本预测重新计算 Accuracy/Macro-F1，确认 BERT 三轮和 64-token 配置。

`自然语言处理作业提交 - 作业一.csv` 中没有查到该学号/姓名。不要覆盖他人行，应在课程指定在线表中填写本人姓名、学号和最终 GitHub 链接。本项目不会自动向他人发送消息或代填课程在线表。

## 大作业阶段文件

目录 `NLP2026_TeamXX_Demo/` 已按英文名组织：

| 阶段 | 文件 | 要求 |
|---|---|---|
| Phase1 | `TeamXX_Proposal.pdf` | 2 页；截止 2026-10-12 |
| Phase2 | `TeamXX_Milestone.pdf` | 2–3 页；当前实现快照，正式中期前更新 |
| Phase2 | `TeamXX_Milestone_Slides.pdf` | 不超过 5 页 |
| Final | `TeamXX_Final_Report.pdf` | ACL 官方模板，正文 8 页，参考文献另页 |
| Final | `TeamXX_Final_Slides.pdf` | 按八分钟展示准备 |
| Final | `TeamXX_Contributions.pdf` | 个人分工与第三方/AI 使用声明 |
| Final | `TeamXX_Code.zip` | 源码、依赖与必要的小型模型、数据 |

`TeamXX` 是显式占位符，必须等正式团队编号分配后统一替换，不能自行编造 Team01。课程未给出的中期、最终截止日期保留为待定。最终 Slides 在本人展示日前一天 23:59 提交。

课程要求上传指定飞书 Course Project 文件夹；本地资料未提供具体链接，因此尚未在线提交。可运行 Demo 已满足“Demo 或视频”的选项，本次无需另制视频。

## 本地整包

`scripts/package_submission.py` 只在七组实测完整后生成 `submission/NLP2026_2312260_TanYi.zip` 和项目源码压缩包。不会把原始提交登记表、代理配置、账号信息、虚拟环境或大型 BERT/GloVe 权重打入压缩包。

整包含课程提供的数据和小型词袋模型，供课程内部复核；公开发布前应确认课程数据的分发要求。BERT 与 GloVe 提供官方名称、revision/哈希及下载代码即可复现，大型模型无需上传。

## 本人最后复核

阅读 `DEFENSE_GUIDE.md`，至少运行一次 Demo 和测试，核对所有材料使用正确姓名与学号，掌握核心公式和局限。AI 辅助已如实披露，不把工具辅助包装成未经辅助的人工工作。
