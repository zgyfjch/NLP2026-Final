"""English-only course-project PDFs. Homework 1 remains a separate report."""
from pathlib import Path
import json
from build_pdfs import ROOT,OUT,TEAM,p,page,table,PageBreak,Paragraph,ParagraphStyle,colors,canvas,html
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.pagesizes import A4

def build(path,flow):
    path.parent.mkdir(parents=True,exist_ok=True)
    def footer(c,d):
        c.setFont('CN',8);c.setFillColor(colors.HexColor('#657386'))
        c.drawString(44,24,'Yi Tan 2312260 | NLP 2026')
        c.drawRightString(A4[0]-44,24,str(d.page))
    SimpleDocTemplate(str(path),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=38,bottomMargin=42,title=path.stem,author='Yi Tan 2312260').build(flow,onFirstPage=footer,onLaterPages=footer)

def result_rows():
    rows=[['Method','Accuracy','Macro-F1','Runtime (s)']]
    for name in ['binary','frequency','tfidf','glove','w2v_ag','w2v_nyt','bert']:
        path=OUT/'results'/f'{name}.json'
        if path.exists():
            r=json.loads(path.read_text(encoding='utf-8'))
            rows.append([name,f"{r['test']['accuracy']:.4f}",f"{r['test']['macro_f1']:.4f}",f"{r['seconds']:.1f}"])
        else:rows.append([name,'Pending','Pending','-'])
    return rows

def metric(name):
    path=OUT/'results'/f'{name}.json'
    if not path.exists(): return 'pending'
    value=json.loads(path.read_text(encoding='utf-8'))['test']
    return f"{value['accuracy']:.2%} accuracy, {value['macro_f1']:.2%} macro-F1"

def proposal():
    flow=[p('NewsLens: An Explainable News Classification System',ParagraphStyle('en-title',fontName='CN',fontSize=20,leading=28,spaceAfter=13)),
          p('Course Project Proposal | Track B: Demonstration'),p('Yi Tan (2312260)')]
    flow+=page('Project overview',[
      'A useful news-routing tool needs more than a predicted topic. Readers should be able to inspect which text features influenced the decision and recognize when a prediction merits manual review. This project studies a compact, locally runnable solution for English sports, politics, and business news. It treats probability quality and explanation validity as separate questions from classification accuracy.',
      'The representation experiments from Homework 1 supply controlled baselines. The course project extends those baselines into an interactive system with calibrated decisions, selective prediction, exact linear explanations, and explicit verification. The aim is a working demonstration with measured behavior, rather than a larger model or an API wrapper.'])
    flow+=page('Problem definition and research questions',[
      'Input: one English news article. Output: class probabilities, a preferred topic, the strongest terms supporting it over the runner-up class, and an accept-or-review decision. The three labels are closed-set topics; the output is not a fact-checking verdict.',
      'RQ1: How do lexical, averaged word-vector, and pretrained neural representations compare under a shared test split? RQ2: Does temperature scaling improve held-out negative log likelihood and calibration error? RQ3: How does a review threshold change accepted coverage and accuracy? RQ4: Can the displayed term contributions exactly reconstruct a class-score margin, and how sensitive are predictions to removing those terms?'])
    flow+=page('System design and original implementation',[
      'The data layer preserves the supplied CSV files, their checksum, and fixed row identifiers. The model layer trains Binary BoW, Word Frequency, TF-IDF, GloVe-100d, two 100-dimensional Word2Vec variants, and the required BERT-base model for offline comparison. Vocabulary, IDF, NYT embeddings, and scalers are fitted on the training partition only.',
      'The decision layer divides the validation set into disjoint temperature-fitting and threshold-selection subsets. The explanation layer computes x_j(w_aj - w_bj) for the predicted class a versus runner-up b and verifies that all terms plus the intercept reconstruct the score gap. A local Flask interface integrates input validation, model selection, prediction, explanations, review guidance, and recorded experiment results.'])
    flow.append(PageBreak())
    flow+=page('Data and evaluation',[
      'The supplied NYT corpus contains 11,519 labeled articles; AG News supplies 90,000 texts for one Word2Vec experiment. A seed-42 shuffle yields 9,215 training, 1,152 validation, and 1,152 test rows. Every method uses the same saved split. Exact duplicate text and class imbalance are audited, with a supplementary test result excluding articles repeated verbatim in training.',
      'Classification is evaluated with accuracy, macro-F1, per-class precision/recall/F1, and confusion matrices. Confidence is evaluated by negative log likelihood and ten-bin expected calibration error. Selective prediction reports both coverage and accepted accuracy. Explanation tests measure maximum score-reconstruction error and compare deleting the top five positive contributors with deleting the same number of random present features.'])
    flow+=page('Expected outcomes and feasibility',[
      'The expected deliverables are a runnable local demo, full source code and dependency information, test-row predictions and metrics, an ACL-style eight-page final report, slides, and a contribution statement. The project tests a hypothesis about practical transparency; it does not assume that BERT or calibration will necessarily improve every metric.',
      'The implementation uses established open-source libraries for standard algorithms while adding its own data contract, decision policy, explanation logic, system integration, and tests. The local computer has an i7-13700H CPU and 16 GB RAM, with no detected NVIDIA GPU. Neural fine-tuning may require a long CPU run, but the deployed demo can use a compact lexical model. Negative or inconclusive results will be reported directly.'])
    flow+=page('Member responsibilities and timeline',[
      'Yi Tan is the sole member, responsible for requirements, data preparation, modeling, implementation, testing, analysis, writing, and presentation. Codex assists drafting and execution; the final contribution statement will distinguish student work from libraries, pretrained weights, and AI assistance. The student must review and be able to explain the final materials.'])
    flow.append(table([['Stage','Planned work and acceptance criterion'],['Proposal | 12 Oct 2026','Define the problem, architecture, evaluation plan, and feasibility; submit this PDF.'],['Midterm | date to be announced','Show a working system core, real initial results, a 2–3-page report, and up to five slides.'],['Final presentation | date to be announced','Finish comparison and system evaluation; prepare an eight-minute talk and live demo.'],['Final submission | date to be announced','Submit the ACL-style report, slides, source, dependencies, demo, and contribution statement.']],[120,385]))
    flow+=page('References consulted for the proposal',[
      'Devlin et al. 2019, BERT, ACL Anthology N19-1423; Pennington et al. 2014, GloVe, ACL Anthology D14-1162; Mikolov et al. 2013, arXiv:1301.3781; Guo et al. 2017, On Calibration of Modern Neural Networks, PMLR 70.'])
    build(TEAM/'Phase1/TeamXX_Proposal.pdf',flow)

def milestone():
    pr=json.loads((OUT/'project_results.json').read_text(encoding='utf-8'))
    flow=[p('NewsLens: Midterm Technical Milestone',ParagraphStyle('en-midtitle',fontName='CN',fontSize=20,leading=28,spaceAfter=13)),
          p('Actual progress snapshot as of 12 October 2026 | Yi Tan (2312260)')]
    flow+=page('Progress summary',[
      'The data audit, reproducible split, three lexical classifiers, two self-trained Word2Vec comparisons, local web demonstration, TF-IDF temperature scaling, review threshold, additive term explanation, deletion diagnostic, and automated system tests have been implemented. This document is an early snapshot, not a claim that the course-designated midterm date has arrived.',
      'The Flask service processes input locally and provides class probabilities, review advice, and terms supporting the selected class over the runner-up. Three lexical models are selectable online; the other methods are offline comparisons. The interface reads stored experiment metrics rather than displaying handwritten scores.'])
    flow.append(table([[str(x) for x in row] for row in result_rows()],[160,115,115,115]))
    flow+=page('Preliminary system evidence',[
      f"The fitted TF-IDF temperature is {pr['temperature']:.4f}; ten-bin test ECE changes from {pr['calibration']['raw_ece']:.4f} to {pr['calibration']['scaled_ece']:.4f}. The validation-selected threshold is {pr['selected_threshold']:.4f}. The system records per-example predictions, metrics, confusion matrices, hyperparameter trials, and split identifiers so each summary can be checked.",
      'Ten automated tests cover shared split membership, tokenization, invalid and oversized input, out-of-vocabulary handling, probability normalization, exact explanation reconstruction, and successful page and API responses. They verify the implemented workflow; they are not a production security assessment.'])
    flow.append(PageBreak())
    flow+=page('Challenges and interpretation',[
      'The sports class dominates the corpus, so macro-F1 and per-class errors must accompany accuracy. Exact duplicate text occurs across the original train and test split. The main comparison retains the fixed course split, while a supplemental TF-IDF analysis excludes eight repeated test texts.',
      'The neural model has a 64-WordPiece limit, while lexical classifiers can use full articles. A length audit finds that all 1,152 test articles exceed the neural limit, so performance differences cannot be attributed solely to architecture. A local CPU and large public model downloads also slow complete neural experiments.',
      'Keyword contributions are algebraic effects on a linear class-score difference. They can be verified exactly, but are not causal explanations or evidence that an article is true. Calibration and accepted-accuracy estimates remain specific to this dataset and split.'])
    flow+=page('Plan adjustment and next steps',[
      'The initial model comparison has been expanded to include negative log likelihood, expected calibration error, accepted coverage, an exact contribution check, a deletion diagnostic, and duplicate sensitivity. The validation rows were split so fitting the temperature and selecting the threshold use disjoint subsets.',
      'Before the formal milestone and final presentation, update this snapshot with all completed required model results, retest the packaged demo from a clean environment, inspect every PDF page, and prepare a direct eight-minute explanation. Multi-seed, event-grouped, and out-of-domain evaluations are future extensions unless recorded artifacts demonstrate they were run.'])
    flow+=page('Individual contribution and tool use',[
      'Sole member: Yi Tan (2312260). Responsibilities cover the full workflow: problem framing, data, modeling, calibration, explanation, integration, tests, reporting, and presentation. Libraries and pretrained weights provide the standard algorithms. Codex assisted with drafting, execution, debugging, and artifacts. The specific boundary between prior work and project work is documented in the separate contribution statement.'])
    build(TEAM/'Phase2/TeamXX_Milestone.pdf',flow)

def contribution():
    flow=[p('Individual Contribution and Tool Use',ParagraphStyle('en-contitle',fontName='CN',fontSize=20,leading=28,spaceAfter=13)),p('NewsLens | Track B Demonstration | Yi Tan (2312260)')]
    flow+=page('Responsibilities and delivered work',[
      'Yi Tan is the sole team member and is responsible for the project question, data audit, fixed experimental split, model comparisons, calibration and explanation design, local interface, tests, result interpretation, written materials, and presentation. This responsibility statement does not claim that every line of source or prose was authored without assistance.',
      'Verifiable work is stored under src/, tests/, templates/, scripts/, artifacts/, and the stage-specific PDF folders. A training task counts as completed only when its metrics and per-example predictions pass the submission verification script. Future plans are not listed as accomplished results.'])
    flow+=page('Prior resources and project-specific contribution',[
      'Prior resources include course-provided CSV files, Stanford GloVe vectors, Google BERT weights, scikit-learn logistic regression, Gensim Word2Vec, PyTorch, Transformers, Flask, and the unmodified official ACL style. Their underlying algorithms and code remain the work of their respective authors.',
      'Project-specific work comprises the shared data protocol and checksum, disjoint validation calibration and threshold selection, exact relative term contributions, feature-deletion diagnostic, local interactive demo, input handling, tests, per-example predictions, and analysis of validity limits. Public course and news-classification repositories were examined as examples of reproducible packaging and reporting; no third-party assignment code or results were imported into this implementation.'])
    flow+=page('AI assistance and student review',[
      'Codex assisted with reading the requirements, drafting and revising code and English prose, executing experiments, debugging the environment, recording results, and producing the PDF materials. Numerical findings come from actual program output; uncompleted experiments are identified explicitly until run. This document is a disclosure, not a signature or a claim that a future oral presentation has already occurred.',
      'Before submission, the student should review the source and reports, run the demo, and be prepared to explain split leakage controls, logistic regression, averaged embeddings, BERT fine-tuning, temperature scaling, threshold selection, and the contribution equation.'])
    build(TEAM/'Final/TeamXX_Contributions.pdf',flow)

def project_slides():
    pr=json.loads((OUT/'project_results.json').read_text(encoding='utf-8'))
    selected=pr['selective_curve'][-1];e=pr['explanation']
    start=('NewsLens: Explainable News Classification',['Problem: a topic label alone does not tell a user when to trust or review a result.','Pipeline: English article → three class scores → relative terms → accept or manual review.','Yi Tan (2312260). The live demo uses a local lexical model.'])
    protocol=('System and Evaluation Protocol',['11,519 labeled articles; shared train / validation / test counts: 9,215 / 1,152 / 1,152.','Seed 42 and saved row identifiers; vocabulary, IDF, and NYT embeddings use training text only.','Original components: calibration, threshold selection, additive explanation, deletion check, interface, and tests.'])
    preliminary=('Measured Classification Baselines',[f"Binary BoW: {metric('binary')}.",f"Word Frequency: {metric('frequency')}.",f"TF-IDF: {metric('tfidf')}.",'The strong linear baseline motivates separate tests of confidence and explanation quality.'])
    validated=('Calibration and Explanation Tests',[f"Test ECE before / after scaling: {pr['calibration']['raw_ece']:.4f} / {pr['calibration']['scaled_ece']:.4f}.",f"Selected threshold: {selected['coverage']:.2%} coverage; {selected['accuracy']:.2%} accuracy among accepted articles.",f"Top-five feature deletion changes {e['top5_flip_rate']:.2%} of predictions; random deletion changes {e['random5_flip_rate']:.2%}.",'Contributions explain linear scores; they are not causal facts about the news.'])
    challenges=('Challenges and Next Steps',['72 exact duplicate texts; eight test texts appear in training.','Every test article exceeds the BERT 64-WordPiece input limit.','Complete required neural and GloVe records, then refresh the formal milestone.','The individual contribution and Codex assistance are disclosed separately.'])
    mid=[start,protocol,preliminary,validated,challenges]
    final=[start,('Background and Questions',['Can a compact local model classify news and show its own evidence?','Does calibration improve probability quality without changing the predicted class?','What coverage does a review threshold retain, and how can explanation arithmetic be checked?']),protocol,
           ('Seven Representation Experiments',['All completed scores are read from the same 1,152 test rows.','The TF-IDF baseline reaches 99.13% accuracy and 98.03% macro-F1.','Averaged embeddings and BERT differ in training corpora and visible input length.']),
           ('Decision Policy',['Use one half of validation to fit temperature and the other to choose a review threshold.','Keep test labels outside model, checkpoint, temperature, and threshold selection.','Term contribution is the feature value times the difference of two class coefficients.']),validated,
           ('Live Demonstration',['Run Launch_Demo.ps1 and open the local browser page.','Analyze a sports story; compare three lexical models and inspect positive term contributions.','Try an empty or all-unknown input to observe clear validation or review behavior.']),
           ('Limitations',['Sports dominates the data; inspect macro-F1 and minority-class errors.','Removing verbatim train/test duplicates does not remove event-level near duplicates.','All test stories exceed 64 WordPiece positions, so BERT sees only a short prefix.','No cross-domain abstention guarantee or human explanation study is claimed.']),
           ('Conclusion and Contribution',['NewsLens provides a tested local classification, calibration, explanation, and review workflow.','Saved splits and per-example predictions make the scores independently checkable.','Yi Tan is the sole member; libraries, pretrained models, and Codex assistance are disclosed.'])]
    def draw(path,items):
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import landscape,A4
        from reportlab.pdfbase import pdfmetrics
        path.parent.mkdir(parents=True,exist_ok=True)
        c=canvas.Canvas(str(path),pagesize=(960,540));c.setTitle(path.stem);c.setAuthor('Yi Tan 2312260')
        for i,(title,body) in enumerate(items):
            c.setFillColor(colors.HexColor('#f6f8fb'));c.rect(0,0,960,540,fill=1,stroke=0)
            c.setFillColor(colors.HexColor('#163d60'));c.setFont('CN',27);c.drawString(52,470,title)
            y=420
            for line in body:
                q=Paragraph(html.escape(line),ParagraphStyle('en-slide',fontName='CN',fontSize=17,leading=27,textColor=colors.HexColor('#263c50')))
                w,h=q.wrap(845,330);q.drawOn(c,56,y-h);y-=h+19
            if y<50:raise ValueError(f'Overfull slide {title}')
            chart=None
            if title in ['Measured Classification Baselines','Seven Representation Experiments']:chart=OUT/'figures/classification.png'
            if title=='Calibration and Explanation Tests':chart=OUT/'figures/calibration.png'
            if chart and chart.exists():c.drawImage(str(chart),120,43,width=720,height=145,preserveAspectRatio=True,anchor='c')
            c.setFont('CN',11);c.setFillColor(colors.HexColor('#657386'))
            c.drawString(52,24,'Yi Tan 2312260 | NewsLens | Track B');c.drawRightString(908,24,f'{i+1} / {len(items)}');c.showPage()
        c.save()
    draw(TEAM/'Phase2/TeamXX_Milestone_Slides.pdf',mid)
    draw(TEAM/'Final/TeamXX_Final_Slides.pdf',final)

def main():
    proposal();milestone();contribution();project_slides()

if __name__=='__main__':main()
