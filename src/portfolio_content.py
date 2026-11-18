"""Shared portfolio content and renderers: README and site use one evidence source."""
import html
import json
from pathlib import Path

REPO = 'https://github.com/rohaan2802/Student_Bayes'
SITE = 'https://rohaan2802.github.io/Student_Bayes/'
RESOURCE = 'https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/GLM.html'


def icon(name):
    paths = {
        'sun':'<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/>',
        'moon':'<path d="M20.8 13a8.8 8.8 0 1 1-9.8-9.8A7 7 0 0 0 20.8 13Z"/>',
        'menu':'<path d="M3 6h18M3 12h18M3 18h18"/>',
        'close':'<path d="m6 6 12 12M6 18 18 6"/>',
        'up':'<path d="m5 12 7-7 7 7M12 5v15"/>',
        'code':'<path d="m8 6-6 6 6 6m8-12 6 6-6 6m-3-15-2 18"/>',
        'download':'<path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/>',
        'external':'<path d="M14 3h7v7m0-7L10 14M10 3H3v18h18v-7"/>',
        'search':'<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
        'left':'<path d="m14 5-7 7 7 7"/>',
        'right':'<path d="m10 5 7 7-7 7"/>',
        'zoom':'<circle cx="10" cy="10" r="6.5"/><path d="m15 15 6 6M10 7v6m-3-3h6"/>',
    }
    return f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths[name]}</svg>'


def content(metrics):
    lo,hi=metrics['accuracy_wilson_95']
    return [
      ('overview','Overview',[
        ('p','Student_Bayes investigates how recorded study habits relate to quiz performance and student categories. It combines an executed Bayesian study with an independently runnable evaluation pipeline, a portable probability model, and a searchable gallery of real outputs.'),
        ('p','The central finding is a strong association between study hours and quiz score, alongside substantial overlap between predictors. Model uncertainty, leakage prevention, and the limits of a small synthetic dataset are part of the analysis rather than footnotes.'),
        ('list',['150 supplied synthetic observations; five recorded fields; three balanced categories.', 'Eight Bayesian models covering correlation, four regression variants, two group-mean models, and multiclass classification.', 'A reproducible Python application, IDE workspace, exported reports, and a browser-based probability explorer.'])]),
      ('inside',"What’s Inside",[
        ('table',(['Artifact','Location','Purpose'],[
          ['Bayesian study','notebooks/student_models.ipynb','Maintained notebook with model definitions, diagnostics, intervals and saved outputs'],
          ['Enhanced application','src/analyze.py','Validation, train-only evaluation, CLI prediction and portable model export'],
          ['Project configuration','pyproject.toml','Dependencies, optional extras and student-bayes command'],
          ['IDE project','Student_Bayes.code-workspace','Open the project; run analysis, build, preview and tests'],
          ['Portfolio build','src/build_portfolio.py + src/portfolio_content.py','Generate the same 24 figures and shared documentation'],
          ['Evidence exports','reports/','Metrics, row-level probabilities, regression comparison and importance'],
          ['Original assignment','archive/assignment_template.ipynb','Preserved instructions and task context']]))]),
      ('data','Dataset and Inputs',[
        ('p','The original CSV is retained without modifying its values. A zero-based source_row in the prediction export identifies the corresponding data row, excluding the header. The target student_type is never included among predictive features.'),
        ('table',(['Field','Role','Interpretation'],[
          ['study_hours','Predictor / grouped outcome','Recorded study hours'],['sleep_hours','Predictor','Recorded sleep hours'],
          ['social_time','Predictor','Recorded social time'],['quiz_score','Regression target','Continuous recorded quiz score'],
          ['student_type','Classification target / group','Original labels: bad, good, excellent']])),
        ('p','Validation checks exact schema, complete rows, finite numeric values, nonnegative hours, varying predictors, and sufficient cases per class. Missing values and duplicates are counted explicitly. Each supplied category contains 50 rows. No generation procedure, time window for hours, or formal score bounds was supplied.')]),
      ('methods','Methods and Model Breakdown',[
        ('table',(['Model','Specification','Question'],[
          ['Bayesian correlation','Bivariate Normal with Uniform(-1,1) rho','How uncertain is the study-score correlation?'],
          ['Simple Gaussian','Centered study hours; Normal mean and slope priors','How does the expected score vary with study hours?'],
          ['Robust Student-t','Same mean; nu = 2 + Exponential(1/10)','Does a heavier-tailed likelihood change the slope?'],
          ['Quadratic','Centered hours and squared hours','Is there convincing evidence for curvature?'],
          ['Multiple regression','All habits standardized with sample SD','What associations remain when other habits are held fixed?'],
          ['Unpooled group means','Independent category mean priors','How do study hours differ across recorded categories?'],
          ['Hierarchical group means','Shared center and between-group scale','How much information do categories share?'],
          ['Bayesian Softmax','Reference score fixed to zero; Categorical likelihood','Which category has the largest posterior mean probability?']])),
        ('p','The notebook uses 94% highest-density intervals and seed 42. Sampling uses four chains with 1,500 tuning and 1,000 retained draws per chain. Regression bands describe the fitted mean, not the wider distribution of a new student’s score. Bayesian R² in this study describes fit to the observed sample.'),
        ('p','The enhanced application adds frequentist logistic regression, study-only OLS, multiple OLS, Ridge with fixed alpha=1, and mean/prior dummy baselines. These are transparent comparison models; they do not reproduce Bayesian posterior draws.')]),
      ('pipeline','Evaluation Pipeline',[
        ('list',['Validate the CSV and create a seed-42 stratified 120/30 train/test split.',
                 'Fit StandardScaler and multinomial logistic regression inside a single training pipeline.',
                 'Run 5-fold stratified CV repeated three times on the 120 training rows only. Each fold refits its own scaler.',
                 'Calculate permutation importance on validation folds inside training data using the increase in log loss after shuffling each predictor.',
                 'Score the reserved 30-row holdout once per fixed model: accuracy, log loss, multiclass Brier score and class-wise precision/recall.',
                 'Fit regression comparisons on the same training rows and report holdout RMSE, MAE and R². Keep descriptive full-sample OLS separate.',
                 'Export numeric coefficients, training mean/scale, source-row predictions and an environment/data fingerprint.']),
        ('code',('python','pipeline = make_pipeline(\n    StandardScaler(),\n    LogisticRegression(max_iter=2000),\n)\npipeline.fit(X_train, y_train)\nprobabilities = pipeline.predict_proba(X_test)')),
        ('p','No hyperparameter search is performed on the holdout. Cross-validation folds overlap across repeats, so their standard deviation is descriptive variation, not a confidence interval. Permutation importance can be shared or diluted by correlated predictors.')]),
      ('results','Results and Interpretation',[
        ('table',(['Measure','Value','Evidence / scope'],[
          ['Logistic holdout accuracy',f"{metrics['baseline_accuracy']:.1%} (30/30)",'Fresh baseline; 10 cases per category'],
          ['95% Wilson interval',f'{lo:.1%} to {hi:.1%}','Binomial interval conditional on the small test sample'],
          ['Training-only repeated CV',f"{metrics['cv_accuracy_mean']:.1%}; SD {metrics['cv_accuracy_sd']:.1%}",'15 validation scores; reserved test rows excluded'],
          ['Prior-only dummy accuracy',f"{metrics['dummy_accuracy']:.1%}",'Same holdout; class-prior baseline'],
          ['Holdout log loss',f"{metrics['baseline_log_loss']:.4f}",'Lower is better; evaluates probability quality'],
          ['Multiclass Brier score',f"{metrics['multiclass_brier']:.4f}",'Mean sum of squared class-probability errors; range 0–2'],
          ['Descriptive OLS R²',f"{metrics['descriptive_ols_r2']:.3f}",'Full-sample fit, not held-out performance'],
          ['Saved Bayesian rho','0.937; 94% HDI 0.917–0.954','Embedded notebook posterior'],
          ['Saved multiple Bayesian R²','0.909','Embedded notebook; in-sample'],
          ['Saved Bayesian classifier','30/30 correct','Embedded notebook; separate model']])),
        ('table',(['Regression holdout model','RMSE','MAE','R²'],[[r['model'],f"{r['rmse']:.3f}",f"{r['mae']:.3f}",f"{r['r2']:.3f}"] for r in metrics['regression_holdout']])),
        ('p','Study hours correlate with quiz score at r=0.939; study and social time correlate at r=-0.936. In the saved Bayesian study, the study slope changes from 0.954 points/hour in the simple model to 0.507 after adjustment. This change and the wider adjusted interval illustrate why marginal associations cannot be read as independent effects.'),
        ('p','Perfect classification within this synthetic dataset is evidence of strong separation in these rows. It is not evidence of error-free prediction for real students. The Wilson interval and baseline comparisons provide context, while the saved posterior checks address different questions.')]),
      ('explorer','Probability Explorer',[
        ('p','Enter study, sleep and social hours to inspect the fixed training-partition logistic baseline. The website applies the exported training scaler, coefficients and numerically stable Softmax entirely in the browser. No entries are sent to a server or saved.'),
        ('p','This explorer uses frequentist baseline probabilities, not Bayesian posterior predictions or a calibrated student assessment. Values outside observed training ranges are flagged as extrapolation. Use the sample presets for a compact demonstration.'),
        ('code',('sh','python src/analyze.py --predict 5 7 3'))]),
      ('source','Source and Project Files',[
        ('p','The enhanced application lives alongside the original Bayesian notebook. All source needed to reproduce the baseline, reports, gallery and webpage is kept in this repository. The workspace provides build and test tasks; pyproject.toml records dependency groups and the CLI entry point.'),
        ('links',[('Open IDE project','Student_Bayes.code-workspace'),('Application source','src/analyze.py'),('Bayesian notebook','notebooks/student_models.ipynb'),('Project configuration','pyproject.toml'),('Portfolio builder','src/build_portfolio.py'),('Shared content','src/portfolio_content.py')]),
        ('code',('text','Student_Bayes/\n├── pyproject.toml                  # project metadata and command\n├── Student_Bayes.code-workspace    # IDE tasks\n├── data/Syn_student.csv            # original 150-row dataset\n├── notebooks/student_models.ipynb  # Bayesian study with saved outputs\n├── src/\n│   ├── analyze.py                  # enhanced runnable application\n│   ├── build_portfolio.py          # evidence figure generation\n│   └── portfolio_content.py        # shared README/site content\n├── reports/                        # metrics, predictions, model and checks\n├── assets/                         # themes, controls and 24 figure pairs\n├── tests/                          # correctness and browser checks\n├── docs/                           # methodology, data and verification\n└── archive/                        # original assignment template'))]),
      ('setup','Setup and Run',[
        ('p','Use Python 3.12 or newer for the baseline. Open Student_Bayes.code-workspace in VS Code, select the environment, then use the Run analysis task. The direct commands below also work from a terminal; no IDE is required.'),
        ('code',('powershell','python -m venv .venv\n.venv\\Scripts\\Activate.ps1\npython -m pip install -r requirements.txt\npython src/analyze.py\npython src/analyze.py --predict 5 7 3\npython -m unittest discover -s tests')),
        ('p','For the optional named command, install this checkout in editable mode: python -m pip install -e . and run student-bayes. Editable mode keeps the command connected to the repository dataset and reports. For a regular package installation, supply --data and --output explicitly. On macOS/Linux, activate the environment with source .venv/bin/activate.'),
        ('code',('powershell','python -m pip install -r requirements-bayesian.txt\npython -m jupyterlab notebooks/student_models.ipynb\n# Run all cells from the repository or notebook directory.')),
        ('p','The saved notebook records Python 3.12, PyMC 5.28.5 and ArviZ 0.23.4. Use an isolated compatible PyMC 5 / ArviZ 0.x environment for resampling. Nutpie is optional. A Bayesian run is substantially slower than the baseline; saved outputs are included for inspection.')]),
      ('outputs','Saved Outputs and Demo',[
        ('table',(['Output','Contents'],[
          ['reports/baseline_metrics.json','Holdout/CV metrics, confusion labels, regression comparisons and correlations'],
          ['reports/baseline_predictions.csv','Source row, three probabilities, actual/predicted labels and confidence'],
          ['reports/demo_model.json','Training-only numeric parameters used by the browser explorer'],
          ['reports/feature_importance.csv','Train-fold permutation importance: mean and spread'],
          ['reports/regression_comparison.csv','Held-out RMSE, MAE and R² for four models'],
          ['reports/group_summary.csv','Group mean, sample SD and median for numeric fields'],
          ['reports/baseline_environment.json','Python/package versions and SHA-256 dataset fingerprint'],
          ['assets/gallery.json','One ordered source for the 24 figure paths and captions']])),
        ('code',('sh','python src/analyze.py\npython src/build_portfolio.py\npython -m http.server 8000\n# Open http://localhost:8000/')),
        ('list',['Inspect the baseline and held-out regression tables, then download the row-level predictions.', 'Try sample presets in the explorer; compare the class probabilities and extrapolation warning.', 'Search the figure gallery for regression; open a figure and use arrow keys, Previous/Next, zoom or swipe.', 'Open the notebook to inspect priors, trace plots, posterior intervals and written interpretation.']),
        ('links',[('Metrics JSON','reports/baseline_metrics.json'),('Prediction CSV','reports/baseline_predictions.csv'),('Regression CSV','reports/regression_comparison.csv'),('Importance CSV','reports/feature_importance.csv'),('Original data','data/Syn_student.csv')])]),
      ('verification','Verification',[
        ('p','Correctness checks cover repeatable predictions, split disjointness, rejection of malformed data, train-only preprocessing, exported-model probability parity and uncertainty intervals. Browser checks exercise both themes, responsive grids, mobile navigation, gallery search/filtering, viewer controls, explorer predictions and Back to Top.'),
        ('p','Saved Bayesian outputs remain identified as saved evidence. A fresh full Bayesian run has not been verified in the current environment; baseline checks do not certify posterior sampling. The detailed verification report records tested scope and limitations.'),
        ('code',('sh','python -m unittest discover -s tests\npython -m pip install "playwright>=1.50,<2"\n# Microsoft Edge must be installed; the check starts its own local server.\npython tests/check_portfolio.py')),
        ('links',[('Verification report','docs/VERIFICATION.md'),('Methodology','docs/METHODOLOGY.md'),('Data dictionary','docs/DATA.md'),('Provenance','docs/PROVENANCE.md')])]),
      ('limitations','Limitations and Responsible Use',[
        ('list',['The data are synthetic and small. Strong class separation and balanced groups may not occur in real cohorts.', 'There are only 30 held-out observations. Perfect accuracy and repeated training-fold scores do not prove external generalization or probability calibration.', 'Correlated study and social hours make attribution unstable. Coefficients and permutation importance describe the fitted models, not causal effects.', 'Regression priors, likelihoods and score bounds can affect conclusions. Gaussian models allow outcomes beyond observed limits.', 'Only three groups inform the hierarchical model; partial pooling is slight with 50 records per group.', 'Generation details and a redistribution license were not supplied for the course data/material. No license is asserted on their behalf.', 'Do not use this academic demo to rank, label or make decisions about real students.'])]),
      ('author','About the Author',[
        ('p','Mohammad Rohaan · FAST-NUCES · MT2002 Statistical Modeling · Fall 2026. This project demonstrates Bayesian reasoning, reproducible evaluation, clear evidence presentation and practical Python tooling.'),
        ('p','The original notebook discloses OpenAI Codex assistance with drafting, execution and review. Results are tied to the supplied dataset and the methods shown in the source.'),
        ('links',[('GitHub profile','https://github.com/rohaan2802')])]),
    ]


def render_block(block, mode):
    kind,value=block
    # HTML escapes all dynamic prose; executable content is authored locally.
    esc=html.escape
    if kind=='p': return value+'\n\n' if mode=='md' else '<p>'+esc(value)+'</p>'
    if kind=='list': return '\n'.join('- '+v for v in value)+'\n\n' if mode=='md' else '<ul>'+''.join('<li>'+esc(v)+'</li>' for v in value)+'</ul>'
    if kind=='code':
        language,text=value
        return f'```{language}\n{text}\n```\n\n' if mode=='md' else f'<div class="code-block"><span class="code-label">{esc(language)}</span><pre><code>{esc(text)}</code></pre><button class="copy-code" type="button" aria-label="Copy code">Copy</button></div>'
    if kind=='links':
        return ' · '.join(f'[{a}]({b})' for a,b in value)+'\n\n' if mode=='md' else '<div class="resource-links">'+''.join(f'<a href="{esc(b)}">{esc(a)}{icon("external")}</a>' for a,b in value)+'</div>'
    if kind=='table':
        headers,rows=value
        if mode=='md':return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(r)+' |\n' for r in rows)+'\n'
        return '<div class="table-scroll" tabindex="0" role="region" aria-label="Scrollable data table"><table><thead><tr>'+''.join('<th scope="col">'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    raise ValueError(kind)


def write_pages(root,manifest,metrics):
    sections=content(metrics)
    badges=[('Python','3.12+','3776AB','python',REPO),('Models','PyMC','176c64','pymc',RESOURCE),('Data','150 synthetic rows','826132','databricks','data/Syn_student.csv'),('Arrays','NumPy','426784','numpy','src/analyze.py'),('Tables','pandas','53617d','pandas','reports/group_summary.csv'),('Evaluation','scikit-learn','956025','scikitlearn','reports/baseline_metrics.json'),('Notebook','Jupyter','92571b','jupyter','notebooks/student_models.ipynb')]
    from urllib.parse import quote
    badge_data=[(f'https://img.shields.io/badge/{quote(a)}-{quote(b).replace("-","--")}-{c}?style=for-the-badge&logo={logo}&logoColor=white',f'{a}: {b}',target) for a,b,c,logo,target in badges]
    opening = '<div align="center">\n\n# Student_Bayes\n\n**Bayesian modeling of student habits and performance, with reproducible evaluation and a visual evidence portfolio.**\n\n'
    opening += '\n'.join(f'[![{label}]({url})]({target})' for url,label,target in badge_data)
    readme = opening + '\n\n</div>\n\n---\n\n## Live Links\n\n| What | Link |\n|---|---|\n'
    readme += f'| Repository | [{REPO}]({REPO}) |\n| GitHub Pages | [{SITE}]({SITE}) |\n| Project resource | [{RESOURCE}]({RESOURCE}) |\n\n---\n\n## Contents\n\n'
    main_sections=[section for section in sections if section[0]!='author']
    author=next(section for section in sections if section[0]=='author')
    contents=[('live-links','Live Links')]+[(sid,title) for sid,title,_ in main_sections]+[('figures','Figures'),('author','About the Author')]
    readme += '\n'.join(f'{i}. [{title}](#{sid})' for i,(sid,title) in enumerate(contents,1))+'\n\n---\n\n'
    for sid,title,blocks in main_sections:
        readme+=f'<a id="{sid}"></a>\n## {title}\n\n'+''.join(render_block(b,'md') for b in blocks)
    readme+='## Figures\n\n'+ ' · '.join(f'[{m["id"]:02d}](#figure-{m["id"]:02d})' for m in manifest)+'\n\n'
    for m in manifest:
        readme+=f'<a id="figure-{m["id"]:02d}"></a>\n### {m["id"]:02d}. {m["title"]}\n\n<p align="left"><a href="{m["full"]}"><img src="{m["preview"]}" alt="{m["title"]}" width="960" height="600"></a></p>\n\n{m["caption"]}\n\n'
    readme+='---\n\n<a id="author"></a>\n## About the Author\n\n'+''.join(render_block(b,'md') for b in author[2])
    (root/'README.md').write_text(readme,encoding='utf-8')
    nav_ids={'overview','data','methods','pipeline','results','explorer','source','setup','outputs','verification'}
    short={'overview':'Overview','data':'Data','methods':'Models','pipeline':'Pipeline','results':'Results','explorer':'Explorer','source':'Source','setup':'Setup','outputs':'Outputs','verification':'Checks'}
    nav=''.join(f'<a href="#{sid}">{short[sid]}</a>' for sid,_,_ in sections if sid in nav_ids)+'<a href="#gallery">Figures</a>'
    article=''
    author_section=''
    for number,(sid,title,blocks) in enumerate(sections,1):
        if sid=='author':
            author_section='<section id="author" class="content-section author-section"><div class="section-heading"><span>14</span><h2>About the Author</h2></div><div class="section-content">'+''.join(render_block(b,'html') for b in blocks)+'</div></section>'
            continue
        article+=f'<section id="{sid}" class="content-section"><div class="section-heading"><span>{number:02d}</span><h2>{title}</h2></div><div class="section-content">'+''.join(render_block(b,'html') for b in blocks)
        if sid=='explorer':article+=explorer(root)
        article+='</div></section>'
    cards=''.join(f'<button class="figure" data-index="{i}" data-category="{m["category"]}" aria-label="Enlarge {m["title"]}"><div class="figure-image"><img src="{m["preview"]}" width="960" height="600" loading="lazy" alt="{m["title"]}"><span class="expand-icon">{icon("zoom")}</span></div><div class="figure-copy"><span class="figure-category">{m["category"]}</span><h3>{m["id"]:02d}. {m["title"]}</h3><p>{m["caption"]}</p></div></button>' for i,m in enumerate(manifest))
    links='<section id="links" class="live-links"><h2>Live Links</h2><div class="table-scroll"><table><thead><tr><th>What</th><th>Link</th></tr></thead><tbody>'+''.join(f'<tr><td>{a}</td><td><a href="{b}">{c} {icon("external")}</a></td></tr>' for a,b,c in [('Repository',REPO,REPO),('GitHub Pages',SITE,SITE),('Project resource',RESOURCE,RESOURCE)])+'</tbody></table></div></section>'
    model=(root/'reports/demo_model.json').read_text(encoding='utf-8')
    page=f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Bayesian student-performance models, a runnable Python project, transparent evaluation and 24 evidence figures."><meta name="author" content="Mohammad Rohaan"><meta name="theme-color" content="#f5f3ed"><title>Student_Bayes | Statistical Modeling</title><link rel="icon" href="assets/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="assets/style.css"><script>try{{document.documentElement.dataset.theme=localStorage.getItem('student-bayes-theme')||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light')}}catch(e){{}}</script></head>
<body><a class="skip" href="#main">Skip to content</a><header><div class="nav-top"><a class="brand" href="#top"><span class="brand-mark">SB</span>Student_Bayes</a><span class="nav-context">A STATISTICAL MODELING STUDY</span><button id="menu" class="icon-button" aria-label="Open menu" aria-controls="nav" aria-expanded="false">{icon('menu')}</button><button id="theme" class="icon-button" aria-label="Toggle color theme">{icon('moon')}</button></div><nav id="nav" aria-label="Main navigation">{nav}</nav></header>
<main id="main"><div id="top" class="hero"><div><p class="eyebrow">ACADEMIC PORTFOLIO / MT2002 / FALL 2026</p><h1>Student habits.<br><em>Measured with uncertainty.</em></h1><p class="lead">From Bayesian reasoning to a runnable Python project. Explore how study, sleep and social time relate to scores—and where the evidence stops.</p><div class="shortcuts"><a class="primary" href="#explorer">Explore the model {icon('right')}</a><a href="Student_Bayes.code-workspace" download>{icon('code')} Open project</a><a href="notebooks/student_models.ipynb" download>{icon('download')} Notebook</a></div></div><aside class="hero-evidence"><p class="eyebrow">THE STUDY AT A GLANCE</p><dl><div><dt>150</dt><dd>Synthetic observations</dd></div><div><dt>8</dt><dd>Bayesian models</dd></div><div><dt>120 / 30</dt><dd>Training / holdout rows</dd></div><div><dt>24</dt><dd>Evidence figures</dd></div></dl><p>Association, uncertainty and honest evaluation.</p></aside></div><div class="badges">{''.join(f'<a href="{target}"><img src="{html.escape(url)}" alt="{label}" height="28"></a>' for url,label,target in badge_data)}</div>{links}{article}
<section id="gallery"><div class="section-heading"><span>13</span><h2>Evidence Gallery</h2></div><p>Figures 01–15 preserve Bayesian notebook outputs. Figures 16–24 show recomputed data, baseline evaluation and relevant source. README and website use these exact files, captions and order.</p><div class="filters"><label>Search figures<div class="search-field">{icon('search')}<input id="search" type="search" placeholder="Regression, probabilities, source…"></div></label><label>Category<select id="filter"><option>All</option>{''.join('<option>'+c+'</option>' for c in sorted(set(m['category'] for m in manifest)))}</select></label><label>Figure background<select id="figure-background"><option value="slate">Slate contrast</option><option value="warm">Warm contrast</option><option value="graphite">Graphite contrast</option></select></label><button id="reset-filters">Reset filters</button></div><p id="count" role="status">24 figures</p><div id="empty" hidden><h3>No matching figures</h3><p>Try a broader search or reset the category filter.</p></div><div class="gallery">{cards}</div></section>{author_section}</main>
<footer><a href="#top">Student_Bayes</a><span>Mohammad Rohaan · FAST-NUCES · Statistical Modeling</span><a href="README.md">Read the documentation</a></footer><button id="top-button" class="icon-button" aria-label="Back to top" hidden>{icon('up')}</button>
<dialog aria-labelledby="viewer-title"><div class="viewer-head"><span id="viewer-position"></span><button id="close" class="icon-button" aria-label="Close figure">{icon('close')}</button></div><h2 id="viewer-title"></h2><div class="zoom-area"><img id="large" alt=""></div><p id="caption"></p><p id="image-error" hidden>Image could not load. Use the full image link or choose another figure.</p><div class="viewer-controls"><button id="previous">{icon('left')} Previous</button><button id="zoom" aria-pressed="false">{icon('zoom')} Zoom</button><a id="original">Full image {icon('external')}</a><button id="next">Next {icon('right')}</button></div><p class="viewer-hint">← / → to navigate · Escape to close · Swipe on touch screens</p></dialog>
<script id="gallery-data" type="application/json">{json.dumps(manifest)}</script><script id="model-data" type="application/json">{model}</script><script src="assets/app.js"></script></body></html>'''
    (root/'index.html').write_text(page,encoding='utf-8')
    (root/'.nojekyll').touch()


def explorer(root):
    model=json.loads((root/'reports/demo_model.json').read_text())
    import pandas as pd
    frame=pd.read_csv(root/'data/Syn_student.csv')
    presets=[]
    for label in ['bad','good','excellent']:
        values=frame[frame.student_type==label][model['features']].median().round(2).tolist()
        presets.append(f'<button type="button" class="preset" data-values="{html.escape(json.dumps(values))}">{label.title()} group median</button>')
    inputs=''.join(f'<label>{name.replace("_"," ").title()}<input id="habit-{i}" name="{name}" type="number" inputmode="decimal" step="0.01" min="0" max="100" required value="{value:.2f}"><small>Training range: {lo:.2f}–{hi:.2f}</small></label>' for i,(name,value,lo,hi) in enumerate(zip(model['features'],[5,7,3],model['observed_min'],model['observed_max'])))
    return '<div class="explorer"><div class="presets">'+''.join(presets)+'</div><form id="predict-form" novalidate><div class="habit-inputs">'+inputs+'</div><button class="primary" type="submit">Calculate probabilities</button><p id="predict-error" role="alert" hidden></p></form><div id="prediction" aria-live="polite"><h3 id="predicted-class"></h3><div id="probability-bars"></div><p id="extrapolation" hidden></p><p class="model-note">Fixed logistic baseline · training rows only · demonstration, not student assessment</p></div></div>'

