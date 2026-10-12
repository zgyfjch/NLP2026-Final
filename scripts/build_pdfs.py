"""Build Chinese submission PDFs directly from recorded experiment JSON."""
from pathlib import Path
import json, sys, html
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts'; TEAM=ROOT/'NLP2026_TeamXX_Demo'
pdfmetrics.registerFont(TTFont('CN',r'C:\Windows\Fonts\msyh.ttc',subfontIndex=0))
STYLE=ParagraphStyle('body',fontName='CN',fontSize=10,leading=16,spaceAfter=8,wordWrap='CJK')
H=ParagraphStyle('h',parent=STYLE,fontSize=14,leading=21,spaceBefore=10,spaceAfter=9,textColor=colors.black)
TITLE=ParagraphStyle('title',parent=STYLE,fontSize=22,leading=31,spaceAfter=15)
SMALL=ParagraphStyle('small',parent=STYLE,fontSize=8,leading=12)
def p(t,style=STYLE): return Paragraph(html.escape(str(t)),style)
def page(title,paras):
    flow=[p(title,H)]
    for t in paras: flow.append(p(t))
    return flow
def table(rows,widths=None):
    t=Table([[p(c,SMALL) for c in r] for r in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6edf4')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#ccd4dd')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)])); return t
def footer(c,d):
    c.setFont('CN',8);c.setFillColor(colors.HexColor('#657386'));c.drawString(44,24,'2312260 谭易 | NLP 2026');c.drawRightString(A4[0]-44,24,str(d.page))
def build(path,flow):
    path.parent.mkdir(parents=True,exist_ok=True); SimpleDocTemplate(str(path),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=38,bottomMargin=42,title=path.stem,author='2312260 谭易').build(flow,onFirstPage=footer,onLaterPages=footer)
def load(name): return json.loads((OUT/name).read_text(encoding='utf-8'))
def result_rows():
    rows=[['方法','Accuracy','Macro-F1','耗时 秒']]
    for name in ['binary','frequency','tfidf','glove','w2v_ag','w2v_nyt','bert']:
        f=OUT/'results'/f'{name}.json'
        if f.exists():
            r=json.loads(f.read_text()); rows.append([name,f"{r['test']['accuracy']:.4f}",f"{r['test']['macro_f1']:.4f}",f"{r['seconds']:.1f}"])
        else: rows.append([name,'尚无完整实测结果','不报告','—'])
    return rows

def proposal():
    f=[p('NewsLens 可解释新闻分类系统',TITLE),p('Course Project Proposal · Track B Demonstration'),p('成员：2312260 谭易　独立成组　团队编号：待分配（TeamXX）')]
    f+=page('背景与动机',[
      '新闻分流不仅需要分类准确，还需要让使用者理解模型依据，并识别需要人工复核的样本。复杂模型未必适合资源有限的个人部署环境；高置信度也不等于预测可靠。本项目围绕英文新闻的体育、政治和商业分类，建立可运行、可解释、可验证的本地演示系统。',
      '作业一的文本表示对比将作为基线。大作业在此基础上新增概率校准、带拒判机制的选择性预测、可加性关键词解释和解释忠实度测试，形成独立于单纯模型训练的系统开发内容。'])
    f+=page('问题定义与项目目标',[
      '输入为一篇英文新闻正文，输出为三类概率、首选类别、相对第二类别的关键词贡献，以及“接受或建议复核”的状态。系统仅处理给定三类新闻，不将低置信度规则宣传为通用的分布外检测器。',
      '核心问题一：在固定数据划分下，词袋表示、平均词向量和 BERT 的分类性能有何差别？核心问题二：温度缩放能否改善概率与正确率的对应关系？核心问题三：提高接受阈值后，覆盖率与接受样本准确率如何变化？核心问题四：解释分数能否重构模型分数差，删除高贡献词的影响是否大于随机删除？'])
    f+=page('技术方案与自主实现',[
      '数据层保留原始 CSV、文件哈希和固定行号划分。建模层实现 Binary BoW、Word Frequency、TF-IDF 与 Logistic Regression，并用 GloVe、AG News Word2Vec、NYT Word2Vec 和 BERT 形成离线对照。',
      '决策层对 TF-IDF 分类器的 logits 做温度缩放；将验证集分成互不重叠的校准与阈值选择子集，测试集不参与调参。解释层以 x_j(w_首选,j−w_次选,j) 计算词语对类别间分数差的贡献。展示层使用 Flask 与原生网页，完成模型选择、预测、解释与实验结果展示。'])
    f.append(PageBreak())
    f+=page('数据与评价计划',[
      '课程提供 NYT 11,519 篇带标签新闻，AG News 90,000 篇文本。NYT 采用 seed=42 随机打乱后按 80%/10%/10% 划分为 9,215/1,152/1,152 篇。词表和 NYT Word2Vec 仅在训练集拟合；AG News 仅用于词向量训练。',
      '分类指标使用 Accuracy、Macro-F1、每类 Precision/Recall/F1 与混淆矩阵；概率指标使用 NLL 和 10 个等宽区间的 ECE；选择性预测报告覆盖率和接受样本准确率。解释验证包括分数重构误差、前五个正贡献特征删除与随机特征删除对照。',
      '数据审计检查缺失值、类别不均衡与重复文本，并补充排除训练集中出现过的测试文本后的评价。运行效率报告 CPU 单篇推理延迟的中位数和 P95；接口测试覆盖空输入、全未登录词、超长文本、非法请求及解释一致性。'])
    f+=page('成员分工与预期成果',[
      '唯一成员谭易负责需求梳理、数据处理、模型训练、系统开发、实验分析、报告撰写及演示答辩。使用 NumPy、scikit-learn、Gensim、PyTorch、Transformers、Flask 等开源库；AI 辅助代码与文稿制作的具体范围在贡献声明中披露，提交前由本人复核并掌握核心方法。',
      '交付内容包括可运行的本地 Demo、完整源代码与环境锁定文件、可追溯的逐样本预测与实验记录、ACL 官方模板的八页最终报告，以及中期和最终汇报 PDF。预期证明系统流程与解释实现正确，不预设任何模型必然胜出。'])
    f+=page('项目时间安排',[])
    f.append(table([['阶段','计划与验收标准'],['Proposal 2026-10-12','完成问题定义、架构与评价协议，提交本提案。'],['中期检查前 以课程通知为准','实现核心模块、完成传统模型与演示流程，形成真实初步结果和不超过五页的汇报。'],['最终展示前','完成 BERT 对照、校准与解释实验，补充错误分析及接口测试，准备八分钟演示。'],['最终提交 以课程通知为准','整理 ACL 八页报告、代码、贡献声明与运行说明；Slides 于本人展示日前一天 23:59 提交。']],[125,380]))
    f+=page('风险与可行性',[
      '本机为 i7-13700H、16 GB 内存，未检测到 NVIDIA GPU。BERT 完整微调可能耗时较长，可将相同代码迁移至 GPU 环境；本地核心 Demo 使用轻量 TF-IDF，不依赖 BERT 在线推理。若校准或删除实验未见改善，将如实报告负面结果与统计局限。'])
    build(TEAM/'Phase1/TeamXX_Proposal.pdf',f)

def homework():
    a=load('data_audit.json'); pr=load('project_results.json'); f=[p('作业一 文本分类实验报告',TITLE),p('2312260 谭易　个人完成　2026年10月12日')]
    f+=page('实验目标与数据协议',[
      '本实验比较三种词袋表示、三种平均词向量表示以及 BERT 微调，在固定 NYT 测试集上报告 Accuracy 和 Macro-F1。全部数字均来自 artifacts/results 中的实测记录；未完成实验不会以推测值代替。',
      '原始要求存在两处不完整：提到“三种 Bag-of-Words”但只列出 Binary 与 Word Frequency，因此补充标准 TF-IDF；“请统一设置”后未给出随机种子，因此固定为 42，并保存完整行号以便复现。',
      'NYT 有 11,519 条，类别为 sports 8,639、politics 1,451、business 1,429；AG News 有 90,000 条。RandomState(42).permutation 随机打乱后，以 floor(0.8N)、floor(0.9N) 切分，训练/验证/测试分别为 9,215/1,152/1,152。未做分层抽样，符合先随机打乱的要求。',
      '传统方法统一将文本转小写，以正则 [a-z]+(?:\'[a-z]+)? 提取英文单词及内部撇号。保留停用词，不做词干化。词袋使用训练集所有出现的词（min_df=1）。测试集不参与词表、IDF、自训练 NYT 词向量、标准化或超参数拟合。'])
    f.append(table([['类别','训练集','验证集','测试集']]+[[k,a['split_counts']['train'].get(k,0),a['split_counts']['val'].get(k,0),a['split_counts']['test'].get(k,0)] for k in ['business','politics','sports']],[150,118,118,119]))
    f+=page('评价指标',[
      'Accuracy 为正确预测数量除以样本数量。对类别 c，Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，F1=2PR/(P+R)；Macro-F1 对全部三类 F1 等权平均。体育类别占比接近四分之三，因此需要同时分析 Macro-F1 和少数类召回率，不能只报告准确率。'])
    f.append(PageBreak())
    f+=page('Task 1 词袋与逻辑回归',[
      'Binary BoW 使用词是否出现的 0/1 向量；Word Frequency 使用原始词频；TF-IDF 使用平滑 IDF：log((1+N)/(1+df))+1，再对文档向量进行 L2 归一化。三者都使用多项逻辑回归、L2 正则化和 lbfgs 优化器。',
      '各方法独立在验证集选择 C∈{0.1,1,10}，按 Macro-F1 最大选取，分数相同时保留较小 C。max_iter=2000，tol=1e-4。模型仅在训练集训练，选出参数后不使用验证集重拟合，因此所有方法训练样本一致。'])
    f+=page('Task 2 平均词向量',[
      'GloVe 使用 Stanford 官方 glove.6B 的 100 维向量。程序仅加载本数据中出现的词以节省内存，不改动预训练参数。AG Word2Vec 在全部 90,000 条 AG 文本上训练；NYT Word2Vec 只在 9,215 条训练文本上训练。',
      'Word2Vec 使用 Skip-gram，vector_size=100、window=5、min_count=2、negative=5、epochs=5、workers=1、seed=42。单线程和 PYTHONHASHSEED=42 用于降低非确定性。有效词向量按出现次数求算术平均；未登录词跳过，全未登录词文档映射为全零向量。',
      '平均向量经仅在训练集拟合的 StandardScaler 标准化，再训练相同逻辑回归并以验证集选择 C。各划分的词覆盖率与零向量文档数保存在对应结果 JSON 中，便于解释语料差异。'])
    f+=page('Task 3 BERT 微调',[
      '使用指定 google-bert/bert-base-uncased 官方预训练权重，分类头输出三类。采用官方 WordPiece tokenizer，padding=max_length、truncation=True、max_length=64，包含特殊 token。所有 BERT 参数参与微调，不冻结编码器，不以更小模型替代。',
      '训练恰好三轮，batch_size=16，AdamW 学习率 2e-5，weight_decay=0.01，10% warmup 后线性衰减，梯度范数裁剪为 1.0。每轮记录验证集 Macro-F1，三轮训练结束后选最佳验证检查点评测测试集。记录种子、运行设备、训练历史及原始模型 revision。'])
    f.append(PageBreak()); f+=page('实测结果汇总',[]); f.append(table(result_rows(),[160,115,115,115]))
    f+=page('验证集选择记录',[])
    tune=[['方法','特征数 / 维数','选定 C','验证 Macro-F1']]
    for method in ['binary','frequency','tfidf','glove','w2v_ag','w2v_nyt']:
        path=OUT/'results'/f'{method}.json'
        if path.exists():
            rr=json.loads(path.read_text());tune.append([method,rr.get('vocabulary_size',rr.get('embedding_dimension')),rr['C'],f"{rr['validation']['macro_f1']:.4f}"])
    f.append(table(tune,[160,115,115,115]))
    f+=page('比较与解释',[
      '词袋模型保留类别相关词的独立权重，适合主题分类。Binary 与 Frequency 的区别说明出现次数并非唯一决定因素；TF-IDF 的归一化可减弱长文档和通用词的影响。模型排名应由表中实测数值判断，不能根据参数规模预先确定。',
      '平均词向量将任意长度文档压缩为 100 维，存储紧凑，但会丢失词序、词间组合及部分稀有类别词。AG 与 NYT 自训练词向量既有语料规模差异，也有领域差异，因此本实验无法把分数差唯一归因为“领域匹配”。',
      '长度审计显示测试新闻平均 818.79 个 WordPiece（含特殊 token），全部 1,152 篇都超过 64。BERT 只读取前 64 个 WordPiece，而传统模型读取完整文本。表示方式和可见文本长度同时变化，不能把模型差异完全归于架构。数据较容易、类别较少时，预训练模型不一定优于线性基线。'])
    f.append(PageBreak());f+=page('类别级错误分析',[])
    r=load('results/tfidf.json'); f.append(table([['真实类别 / 预测','business','politics','sports']]+[[lab,*row] for lab,row in zip(r['labels'],r['test']['confusion_matrix'])],[160,115,115,115]))
    f.append(p('行是真实类别，列是预测类别。少数类之间的误分比 sports 类别更值得关注。完整逐样本预测与每类指标均随代码提供，可按 row_id 回查原文。'))
    f+=page('数据质量与结果边界',[
      f"原始 NYT 有 {a['duplicates']} 条重复文本（按 text 完全相同计数）；验证集有 {a['val_rows_seen_in_train']} 条、测试集有 {a['test_rows_seen_in_train']} 条文本在训练集中出现。主结果保留指定统一划分，补充在 1,144 条未见测试文本上的 TF-IDF Accuracy={pr['unseen_test']['accuracy']:.4f}、Macro-F1={pr['unseen_test']['macro_f1']:.4f}。这项排除不能消除所有近重复或报道事件层面的泄漏。",
      '本次只使用一个随机划分，没有以多次种子重复估计方差。总体准确率较高时，个别错例即可改变排名；不得将千分位差异表述为统计显著。原始数据的新闻来源和时间范围未由课程文件详细说明，不推断其代表全部 NYT 新闻。',
      '训练耗时包含特征构建和调参，不等于纯模型优化时间。CPU 同时承担多个任务时，耗时不能作为严格硬件基准。实验不涉及事实核查、公平性保证或跨领域泛化证明。'])
    f+=page('复现与提交内容',[
      '环境：Windows，Python 3.10，CPU i7-13700H，16 GB RAM。精确包版本见 requirements-lock.txt。README 提供逐步运行命令及一键 PowerShell 入口。',
      'src/common.py 负责固定划分和评价；train_classical.py 负责 Task 1/2；train_bert.py 负责 Task 3。artifacts/split.json 保存数据哈希和划分行号；artifacts/results 保存结果与逐样本预测。报告由 scripts/build_pdfs.py 从结果自动生成。',
      '提交前应核对 BERT 与 GloVe 两行均为完整实测结果、源码可运行、GitHub 链接对助教可访问。原始提交登记表含其他同学个人信息，不属于作业交付内容。'])
    f+=page('引用与工具使用声明',[
      'Mikolov et al. 2013. Efficient Estimation of Word Representations in Vector Space. https://arxiv.org/abs/1301.3781',
      'Pennington, Socher, and Manning. 2014. GloVe: Global Vectors for Word Representation. https://aclanthology.org/D14-1162/',
      'Devlin et al. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. https://aclanthology.org/N19-1423/',
      '课程 CSV 和作业要求为任务来源；scikit-learn、Gensim、PyTorch、Transformers 提供算法库实现。使用 Codex 辅助实现、排查运行问题、整理结果和生成报告；实验数字由实际程序输出，本人提交前需审阅并能够解释全部核心内容。'])
    build(ROOT/'reports/HW1_2312260_TanYi.pdf',f)

def milestones():
    pr=load('project_results.json'); f=[p('NewsLens 中期阶段技术记录',TITLE),p('截至 2026年10月12日的实际进展快照 · 2312260 谭易')]
    f+=page('已完成工作',[
      '已完成需求分析、数据审计、固定划分、三类词袋训练、两类 Word2Vec 训练、本地网页演示、TF-IDF 概率校准、复核阈值选择、可加性解释和自动接口测试。当前记录反映真实实现状态，不代表未来指定中期日期已经到来。',
      '系统采用本地 Flask 服务，浏览器输入英文文本，返回三类概率、复核提示和词语贡献。模型选择支持 Binary、Frequency、TF-IDF；其他表示方法用于离线比较。没有调用外部推理 API。'])
    f.append(table(result_rows(),[160,115,115,115]))
    f+=page('核心模块的可验证证据',[
      '实验结果文件记录 Accuracy、Macro-F1、类别级指标、混淆矩阵、超参数、耗时和逐样本预测。测试文件 tests/test_system.py 验证固定划分、输入校验、概率归一性、解释重构与页面/API 可用性。',
      f"温度缩放参数 T={pr['temperature']:.4f}，在独立验证子集选择阈值 {pr['selected_threshold']:.4f}。测试 ECE 从 {pr['calibration']['raw_ece']:.4f} 变为 {pr['calibration']['scaled_ece']:.4f}。"])
    f.append(PageBreak()); f+=page('初步发现与挑战',[
      '词袋模型在此三分类数据上已较强，因此大作业的主要价值不能仅依赖准确率提升，而应关注校准、选择性预测、可解释性和可复现系统。',
      'sports 类别占比很高，Accuracy 容易掩盖少数类表现。重复文本跨训练与测试出现，需要保留主划分同时给出去重复敏感性分析。',
      '全词袋与截断 BERT 的可见输入不等长，不能直接宣称差异源自模型能力。词语贡献是模型分数的精确代数解释，但并不是文本事实、语言因果或用户理解的证据。',
      '本机没有 NVIDIA GPU，BERT 训练较慢；权重下载需要可用网络。代码支持每轮检查点恢复，核心 Demo 不依赖大型模型权重。'])
    f+=page('计划调整与后续安排',[
      '在原有准确率比较外增加 NLL、ECE、覆盖率与接受准确率，并将验证集分成校准集和阈值选择集，避免两项操作共用同一批验证样本。',
      '后续优先完成所有规定模型的训练与报告同步，核对多模型可比性，再扩展固定多随机种子实验、事件级去重和分布外新闻评测；这些扩展未完成前不计为已取得成果。',
      '正式中期提交前根据课程安排更新这份快照；最终准备八分钟展示和三分钟问答。团队编号由课程正式分配后统一替换 TeamXX。'])
    f+=page('个人贡献与第三方工具',[
      '唯一成员：2312260 谭易。负责该项目全部角色，涵盖数据、建模、系统、验证、报告与展示。AI 辅助工具参与代码和文字起草，第三方库负责底层算法。自主项目内容是面向课程数据的完整流程、决策与解释模块、验证协议和演示系统；不是重新发明底层算法。'])
    build(TEAM/'Phase2/TeamXX_Milestone.pdf',f)
    f=[p('个人贡献与工具使用声明',TITLE),p('NewsLens · Track B · 2312260 谭易 · 独立成组')]
    f+=page('职责与交付范围',[
      '本组仅一名成员，项目责任全部由谭易承担：问题定义、数据质量分析、固定划分、模型对照、校准与解释设计、系统集成、测试、实验分析、报告和课堂展示。该责任划分不应被理解为所有代码和文稿均未经工具辅助。',
      '已形成的可核查材料为 src/、tests/、templates/、scripts/、artifacts/ 和各阶段 PDF。是否完成某项训练以对应结果 JSON 及日志为准；不得把计划任务计为完成。'])
    f+=page('已有成果与项目新增内容',[
      '已有成果：课程提供的数据，Stanford GloVe，Google BERT，scikit-learn Logistic Regression，Gensim Word2Vec，以及 PyTorch、Transformers、Flask 等通用库；ACL 官方样式文件保留来源。',
      '项目新增内容：统一实验协议与数据审计、验证集拆分校准和阈值选择、类别间线性贡献解释、特征删除验证、本地交互页面、输入校验、接口测试、逐样本结果及复现文档。'])
    f+=page('AI 辅助与本人复核',[
      '本项目使用 Codex 协助阅读要求、起草和修改代码、执行实验、排查环境问题、分析记录、生成文稿与排版。所有性能数据必须来自真实运行，引用对应实际公开来源，不以 AI 生成数字替代实验。',
      '本人应在提交前运行示例、审阅所有报告，理解数据泄漏防范、逻辑回归、词向量平均、BERT 微调、温度缩放和解释分数公式，并能在答辩中独立说明。该文档不代替本人的签字，也不宣称尚未发生的审阅或课堂展示已经完成。'])
    build(TEAM/'Final/TeamXX_Contributions.pdf',f)

def slides(path,items):
    path.parent.mkdir(parents=True,exist_ok=True); c=canvas.Canvas(str(path),pagesize=(960,540));c.setTitle(path.stem); c.setAuthor('2312260 谭易')
    for i,(title,body) in enumerate(items):
        c.setFillColor(colors.HexColor('#f6f8fb')); c.rect(0,0,960,540,fill=1,stroke=0)
        c.setFillColor(colors.HexColor('#163d60'));c.setFont('CN',27);c.drawString(52,469,title)
        y=418
        for line in body:
            q=Paragraph(html.escape(line),ParagraphStyle('slide',fontName='CN',fontSize=18,leading=29,textColor=colors.HexColor('#263c50'),wordWrap='CJK'))
            w,h=q.wrap(845,360); q.drawOn(c,56,y-h);y-=h+22
        if y<50: raise ValueError(f'Slide overflow {title}')
        chart=None
        if title in ['初步分类结果','七组模型实测对比']: chart=OUT/'figures/classification.png'
        if title=='校准与解释验证': chart=OUT/'figures/calibration.png'
        if chart and chart.exists():
            c.drawImage(str(chart),130,45,width=700,height=150,preserveAspectRatio=True,anchor='c')
        c.setFont('CN',11);c.setFillColor(colors.HexColor('#657386'));c.drawString(52,24,'2312260 谭易 · NewsLens · Track B');c.drawRightString(908,24,f'{i+1} / {len(items)}');c.showPage()
    c.save()

def make_slides():
    r=load('project_results.json'); selected=r['selective_curve'][-1]; e=r['explanation']
    items=[('NewsLens 可解释新闻分类系统',['问题：不仅输出类别，还要说明模型依据和复核边界。','输入英文新闻 → 三类概率 → 关键词贡献 → 接受或人工复核。','独立成组：2312260 谭易。演示以本机 TF-IDF 分类器为核心。']),
      ('系统实现与实验协议',['NYT 11,519 条，训练 / 验证 / 测试 = 9,215 / 1,152 / 1,152。','共享 seed=42 固定行号；词表、IDF、NYT Word2Vec 只拟合训练集。','自主实现：校准与阈值选择、分数差解释、删除实验、网页和自动测试。']),
      ('初步分类结果',['Binary BoW：Accuracy 98.78%，Macro-F1 97.25%。','Word Frequency：Accuracy 99.05%，Macro-F1 97.74%。','TF-IDF：Accuracy 99.13%，Macro-F1 98.03%。','强线性基线意味着项目价值还需要由系统设计与可靠性分析体现。']),
      ('校准与解释验证',[f"校准前 / 后 ECE：{r['calibration']['raw_ece']:.4f} / {r['calibration']['scaled_ece']:.4f}。",f"选定阈值下覆盖 {selected['coverage']:.2%}，接受样本准确率 {selected['accuracy']:.2%}。",f"删除正贡献前五词翻转率 {e['top5_flip_rate']:.2%}；随机删除 {e['random5_flip_rate']:.2%}。",'解释为线性分数的代数分解，不等于因果解释。']),
      ('挑战与下一步',['审计发现 72 条重复文本，测试集有 8 条在训练集中出现。','本机无 NVIDIA GPU，完整 BERT 对照需要较长计算时间。','正式中期前更新运行快照；后续补充多随机种子与事件级去重评测。','个人负责全流程；开源算法与 AI 辅助范围已单独披露。'])]
    slides(TEAM/'Phase2/TeamXX_Milestone_Slides.pdf',items)
    final=[items[0],('背景与研究问题',['如何在个人电脑部署可解释的新闻分类器？','高准确率下，概率校准与人工复核机制是否仍有价值？','怎样验证关键词解释对应真实模型计算，而不只是看起来合理？']),items[1],items[2],
      ('决策机制与验证隔离',['验证集一半用于拟合温度 T，另一半用于选择接受阈值。','测试集只用于最终评价，不参与选择 C、T 或阈值。','关键词贡献 = TF-IDF 值 × 首选与次选类别的系数差。']),items[3],
      ('演示流程',['启动 Launch_Demo.ps1，在浏览器打开本地系统。','输入体育新闻，比较三种模型，查看概率和关键词贡献。','输入全未登录词或空文本，检查复核和错误提示。','展示实际结果表，并说明模型预测不是事实核查。']),
      ('局限与结果解释',['体育类占比很高，必须结合 Macro-F1 和类别级指标。','去除重复测试文本后 TF-IDF 仍保持较高分数，但未排除近重复事件。','64-token BERT 与全文词袋输入预算不同，不能把差异全归于架构。','选择性预测仅在当前三分类分布上评价，尚无分布外可靠性保证。']),
      ('结论与个人贡献',['完成可运行的本地分类、校准、解释与复核系统。','用逐样本预测、固定数据划分和自动测试提供可复核证据。','2312260 谭易负责全流程；引用第三方算法，披露 Codex 辅助。','答辩重点：数据泄漏、概率校准、解释公式与结果边界。'])]
    final[3]=('七组模型实测对比',['完整结果来自同一组 1,152 条测试新闻，图表直接读取实验记录。','TF-IDF 基线 Accuracy 99.13%，Macro-F1 98.03%；应结合类别级错误解释。','词向量与 BERT 的结果见图，训练语料和可见输入长度不同，不能仅比较模型大小。'])
    slides(TEAM/'Final/TeamXX_Final_Slides.pdf',final)

if __name__=='__main__':
    homework()
    from build_project_en import main
    main()
