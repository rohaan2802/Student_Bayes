"""Build the static portfolio and 24 framed figures from real evidence."""
from pathlib import Path
import base64, hashlib, io, json, textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageOps
from analyze import ROOT, FEATURES, CLASSES, load_data
from portfolio_content import write_pages

def build():
    nbpath = ROOT/'notebooks/student_models.ipynb'
    nb = json.loads(nbpath.read_text(encoding='utf-8'))
    frame = load_data(ROOT/'data/Syn_student.csv')
    result = json.loads((ROOT/'reports/baseline_metrics.json').read_text())
    environment = json.loads((ROOT/'reports/baseline_environment.json').read_text())
    if environment['dataset_sha256'] != hashlib.sha256((ROOT/'data/Syn_student.csv').read_bytes()).hexdigest():
        raise ValueError('Dataset differs from the saved baseline. Run python src/analyze.py before building.')
    predictions = pd.read_csv(ROOT/'reports/baseline_predictions.csv')
    titles = [
      ('Habit and score distributions','EDA','Four numeric distributions from the supplied 150-row dataset.'),
      ('Habits against quiz score','EDA','Saved notebook scatter plots show associations across the three categories.'),
      ('Scores across student categories','EDA','Saved group boxplots compare score distributions.'),
      ('Pearson correlation matrix','EDA','Saved sample correlations reveal strongly overlapping predictors.'),
      ('Bayesian correlation posterior','Bayesian','Saved bivariate-normal posterior for study hours and quiz score; 94% HDI.'),
      ('Simple regression chain diagnostics','Bayesian','Saved density and chain traces for the Gaussian model.'),
      ('Simple regression mean uncertainty','Bayesian','Saved 94% mean HDI; this is not a prediction interval for a new student.'),
      ('Gaussian versus robust regression','Bayesian','Saved Gaussian and Student-t fits give similar study slopes.'),
      ('Linear versus quadratic regression','Bayesian','Saved mean curves illustrate weak evidence for curvature.'),
      ('Multiple regression chain diagnostics','Bayesian','Saved traces include the three standardized predictor slopes.'),
      ('Adjusted coefficient intervals','Bayesian','Saved multiple-regression 94% HDIs in score points per predictor SD.'),
      ('Joint coefficient posterior','Bayesian','Saved posterior relationships show slope trade-offs under collinearity.'),
      ('Posterior predictive checks','Bayesian','Saved replicated-score densities compared with observed scores for four models.'),
      ('Unpooled and hierarchical means','Bayesian','Saved group-mean intervals show limited pooling with 50 rows per category.'),
      ('Bayesian test confusion matrix','Classification','Saved notebook result: 30 of 30 test students classified correctly.'),
    ]
    raw = []
    for c in nb['cells']:
        for o in c.get('outputs', []):
            if 'image/png' in o.get('data', {}):
                raw.append(Image.open(io.BytesIO(base64.b64decode(o['data']['image/png']))).convert('RGB'))
    assert len(raw) == 15
    plt.rcParams.update({'font.size':19,'axes.labelsize':21,'xtick.labelsize':16,'ytick.labelsize':17,'legend.fontsize':16,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#465c65','text.color':'#142c38','axes.labelcolor':'#142c38','xtick.color':'#142c38','ytick.color':'#142c38','axes.prop_cycle':matplotlib.cycler(color=['#166b68','#4969a1','#aa642a'])})
    def plot(title, category, caption, draw):
        fig, ax = plt.subplots(figsize=(12,6.2), layout='constrained')
        draw(ax)
        buffer = io.BytesIO(); fig.savefig(buffer, format='png', dpi=120); plt.close(fig)
        raw.append(Image.open(buffer).convert('RGB')); titles.append((title,category,caption))
    plot('Balanced category counts','EDA','Recomputed counts: each category contains 50 synthetic students.', lambda ax: (ax.bar(CLASSES,frame.student_type.value_counts().reindex(CLASSES),color=['#117a75','#5576aa','#b26929']),ax.set(ylabel='Students',xlabel='Recorded category')))
    plot('Training-only cross-validation','Verification','Fifteen validation-fold accuracies, using only the 120 training rows; the dotted line is the prior-only holdout baseline.',lambda ax:(ax.plot(range(1,16),result['cv_scores'],'o-',color='#166b68'),ax.axhline(result['dummy_accuracy'],linestyle='--',color='#aa642a',label='Prior-only holdout reference'),ax.set(xlabel='Validation fold across 3 repeats',ylabel='Accuracy',ylim=(0,1.08),xticks=[1,5,10,15]),ax.legend(loc='center right')))
    plot('Study hours by category','EDA','Recomputed category means and sample SDs; error bars show spread, not uncertainty in the mean.',lambda ax:(ax.bar(CLASSES,frame.groupby('student_type').study_hours.mean().reindex(CLASSES),yerr=frame.groupby('student_type').study_hours.std().reindex(CLASSES),capsize=8,color='#117a75'),ax.set(ylabel='Study hours: mean ± sample SD')))
    plot('Baseline test confusion matrix','Classification','Fresh frequentist logistic baseline on a stratified 120/30 split; separate from the Bayesian model.',lambda ax:(ax.imshow(result['baseline_confusion'],cmap='Blues',vmin=0,vmax=10),ax.set(xticks=range(3),yticks=range(3),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='Actual'),[ax.text(j,i,str(v),ha='center',va='center',color='white' if v>=5 else '#17303c',fontsize=22) for i,row in enumerate(result['baseline_confusion']) for j,v in enumerate(row)]))
    plot('Baseline test class probabilities','Classification','Fresh baseline probabilities for 30 held-out rows; each stacked bar sums to one.',lambda ax:([ax.bar(range(30),predictions[k],bottom=predictions[CLASSES[:i]].sum(axis=1) if i else None,label=k) for i,k in enumerate(CLASSES)],ax.set(xlabel='Test row in saved prediction order',ylabel='Probability',ylim=(0,1)),ax.legend()))
    from sklearn.linear_model import LinearRegression
    fit=LinearRegression().fit(frame[FEATURES],frame.quiz_score); fitted=fit.predict(frame[FEATURES]); residual=frame.quiz_score-fitted
    plot('Held-out regression comparison','Baseline','RMSE on the same 30 held-out rows; every model is fitted on the 120 training rows. Lower is better.',lambda ax:(ax.barh([r['model'] for r in result['regression_holdout']],[r['rmse'] for r in result['regression_holdout']],color=['#a1aaa9','#4969a1','#166b68','#aa642a']),ax.set(xlabel='Held-out RMSE (quiz-score points)')))
    plot('OLS residual pattern','Baseline','Fresh full-sample residuals help inspect departures from a linear mean.',lambda ax:(ax.scatter(fitted,residual,color='#117a75'),ax.axhline(0,color='#b26929'),ax.set(xlabel='Fitted quiz score',ylabel='Observed − fitted score')))
    importance=pd.read_csv(ROOT/'reports/feature_importance.csv')
    plot('Train-fold permutation importance','Verification','Validation-fold log-loss increase after shuffling each feature; bars show mean and SD across 50 permutations, not causal effects.',lambda ax:(ax.barh(importance.feature,importance.mean_log_loss_increase,xerr=importance.std_log_loss_increase,color='#166b68',capsize=5),ax.set(xlabel='Increase in validation log loss')))
    from inspect import getsource
    from analyze import make_classifier, split_rows
    code = getsource(make_classifier) + "\n" + getsource(split_rows)
    def code_panel(ax):
        ax.axis('off')
        ax.text(.015,.98,code,transform=ax.transAxes,va='top',fontfamily='monospace',fontsize=18,color='#142c38',linespacing=1.7)
    plot('Training pipeline source','Source','Actual src/analyze.py code: the scaler stays inside the estimator, and the stratified split uses a fixed seed.',code_panel)
    def font(size, bold=False):
        candidates=[Path('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
        for path in candidates:
            if path.exists():return ImageFont.truetype(str(path),size)
        return ImageFont.load_default(size=size)
    heading=font(32,True); small=font(20); caption_font=font(23)
    manifest=[]
    for i,(im,(title,category,caption)) in enumerate(zip(raw,titles),1):
        canvas=Image.new('RGB',(1440,900),'#d7e3e6'); d=ImageDraw.Draw(canvas)
        d.rectangle((0,0,1440,86),fill='#183541')
        d.text((30,8),f'{i:02d}  /  {title}',font=heading,fill='#ffffff')
        source='SAVED BAYESIAN NOTEBOOK' if i<=15 else 'RECOMPUTED DATA / BASELINE / SOURCE'
        d.text((32,49),f'STUDENT_BAYES  /  {category.upper()}  /  {source}',font=small,fill='#c1e1d9')
        # Trim blank margins, then scale both small and large sources to the same
        # plotting area. Aspect ratio is preserved; chart content is never cropped.
        im=im.convert('RGB')
        difference=ImageChops.difference(im,Image.new('RGB',im.size,'white')).convert('L')
        bounds=difference.point(lambda v:255 if v>18 else 0).getbbox()
        if bounds:im=im.crop((max(0,bounds[0]-14),max(0,bounds[1]-14),min(im.width,bounds[2]+14),min(im.height,bounds[3]+14)))
        im=ImageOps.contain(im,(1408,714),Image.Resampling.LANCZOS)
        # ImageOps.contain supports enlargement; white plotting surface provides
        # constant contrast in both README and light/dark website themes.
        d.rounded_rectangle((8,90,1432,824),radius=10,fill='white',outline='#879da5',width=2)
        canvas.paste(im,((1440-im.width)//2,100+(714-im.height)//2))
        d.text((30,837),'\n'.join(textwrap.wrap(caption,105)),font=caption_font,fill='#183541',spacing=6)
        slug=f'{i:02d}-'+title.lower().replace(' ','-')+'-v5.png'
        full='assets/figures/'+slug; preview='assets/previews/'+slug
        canvas.save(ROOT/full,optimize=True)
        canvas.resize((960,600),Image.Resampling.LANCZOS).save(ROOT/preview,optimize=True)
        manifest.append(dict(id=i,title=title,category=category,caption=caption,full=full,preview=preview))
    (ROOT/'assets/gallery.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    write_pages(ROOT,manifest,result)


if __name__ == "__main__":
    build()
