"""Self-contained asset and Edge-browser checks for the static portfolio."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import sys
from pathlib import Path
from threading import Thread
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from analyze import predict_parameters
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--capture',action='store_true',help='Save temporary visual review captures in .review/')
args=parser.parse_args()
data=json.loads((ROOT/'assets/gallery.json').read_text())
readme=(ROOT/'README.md').read_text(encoding='utf-8')
page=(ROOT/'index.html').read_text(encoding='utf-8')
model=json.loads((ROOT/'reports/demo_model.json').read_text())
assert len(data)==24
assert len(list((ROOT/'assets/figures').glob('*.png')))==24
assert len(list((ROOT/'assets/previews').glob('*.png')))==24
for m in data:
    assert Image.open(ROOT/m['full']).size==(1440,900)
    assert Image.open(ROOT/m['preview']).size==(960,600)
    assert m['caption'] in readme and m['caption'] in page
    assert m['full'] in readme and m['preview'] in readme
assert re.findall(r'<img src="(assets/previews/[^\"]+)"',readme)==[m['preview'] for m in data]
ids=set(re.findall(r'id="([^\"]+)"',page))
for href in re.findall(r'(?:href|src)="([^\"]+)"',page):
    if href.startswith('#'):
        assert href[1:] in ids,href
    elif not href.startswith(('http','data:')):
        assert (ROOT/href).is_file(),href
assert len(re.findall(r'\|.*\]\(https://',readme.split('## Live Links')[1].split('## Contents')[0]))==3
assert len(re.findall(r'img.shields.io',readme))==7
assert readme.startswith('<div align="center">')
assert re.findall(r'^## (.+)$',readme,re.M)[-1]=='About the Author'
assert readme.count('## About the Author')==1
assert readme.rfind('<a id="author"></a>')>readme.rfind('<a id="figure-24"></a>')
assert readme.index('</div>') < readme.index('## Live Links')
assert len(re.findall(r'^\d+\. \[',readme.split('## Contents')[1].split('---')[0],re.M))==15
for url in ['https://github.com/rohaan2802/Student_Bayes','https://rohaan2802.github.io/Student_Bayes/','https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/GLM.html']:
    assert f'[{url}]({url})' in readme
assert not re.search(r'planned publication|reserved for|ready for review',readme+page,re.I)

class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass

server=ThreadingHTTPServer(('127.0.0.1',0),partial(QuietHandler,directory=str(ROOT)))
Thread(target=server.serve_forever,daemon=True).start()
base=f'http://127.0.0.1:{server.server_port}/'
out=[]
try:
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge')
        for width,height in [(1440,1000),(820,1180),(375,812)]:
            for theme in ['light','dark']:
                context=browser.new_context(viewport={'width':width,'height':height},color_scheme=theme,has_touch=True)
                tab=context.new_page();errors=[]
                tab.on('pageerror',lambda e:errors.append(str(e)))
                tab.goto(base,wait_until='networkidle')
                assert tab.evaluate('document.documentElement.scrollWidth<=innerWidth'),(width,theme)
                assert tab.locator('.gallery').evaluate('(e)=>getComputedStyle(e).gridTemplateColumns.split(" ").length')==(3 if width>1100 else 2 if width>600 else 1)
                assert tab.locator('html').get_attribute('data-theme')==theme
                assert tab.locator('nav a').count()==12
                author_link=tab.locator('nav a').last
                assert author_link.inner_text()=='Author'
                assert author_link.get_attribute('href')=='#author'
                if width<=1100:
                    tab.locator('#menu').click()
                author_link.click()
                tab.wait_for_function("location.hash === '#author' && Math.abs(document.querySelector('#author').getBoundingClientRect().top) < innerHeight")
                assert tab.locator('#menu').get_attribute('aria-expanded')=='false'
                tab.goto(base,wait_until='networkidle')
                assert tab.locator('main section').last.get_attribute('id')=='author'
                assert tab.locator('nav a').first.evaluate('(e)=>parseFloat(getComputedStyle(e).fontSize)')>=18
                assert tab.locator('#top-button').evaluate('(e)=>parseFloat(getComputedStyle(e).width)')>=56
                tab.locator('#figure-background').select_option('graphite')
                assert tab.locator('.gallery').get_attribute('data-background')=='graphite'
                tab.locator('#figure-background').select_option('warm')
                assert tab.locator('.gallery').get_attribute('data-background')=='warm'
                tab.locator('#figure-background').select_option('slate')
                tab.locator('#theme').click();assert tab.locator('html').get_attribute('data-theme')!=theme
                assert tab.locator('#theme svg').count()==1
                tab.locator('#theme').click()
                if width<=1100:
                    assert tab.locator('#menu svg').is_visible()
                    tab.locator('#menu').click();assert tab.locator('#menu').get_attribute('aria-expanded')=='true'
                    tab.locator('nav a').first.click();assert tab.locator('#menu').get_attribute('aria-expanded')=='false'
                # Browser prediction is numerically identical to the Python path.
                actual=tab.locator('.prob-value').evaluate_all('(nodes)=>nodes.map(n=>Number(n.dataset.probability))')
                expected=predict_parameters(model,[5,7,3])
                assert all(abs(a-expected[k])<1e-12 for a,k in zip(actual,model['classes']))
                for button in tab.locator('.preset').all():
                    button.click();assert tab.locator('.prob-row').count()==3
                tab.locator('#habit-0').fill('99');tab.locator('#predict-form button[type=submit]').click()
                assert tab.locator('#extrapolation').is_visible()
                # Validate custom inline error path without native form interception.
                tab.locator('#habit-0').fill('-1');tab.locator('#predict-form button[type=submit]').click()
                assert tab.locator('#predict-error').is_visible()
                assert not tab.locator('#prediction').is_visible()
                tab.locator('#habit-0').fill('5');tab.locator('#predict-form').dispatch_event('submit')
                assert not tab.locator('#predict-error').is_visible()
                assert tab.locator('#prediction').is_visible()
                tab.locator('#search').fill('quadratic');assert tab.locator('.figure:visible').count()==1
                tab.locator('#search').fill('nonsense-no-match');assert tab.locator('.figure:visible').count()==0
                assert tab.locator('#empty').is_visible()
                tab.locator('#reset-filters').click();assert tab.locator('.figure:visible').count()==24
                tab.locator('#filter').select_option('Bayesian');assert tab.locator('.figure:visible').count()==10
                tab.locator('.figure:visible').first.click()
                tab.locator('#previous').click();assert tab.locator('#viewer-title').inner_text().startswith('14.')
                tab.keyboard.press('Escape')
                tab.locator('#filter').select_option('All');tab.locator('.figure').first.click()
                assert tab.locator('dialog').is_visible()
                tab.keyboard.press('ArrowRight');assert tab.locator('#viewer-title').inner_text().startswith('02.')
                tab.locator('#previous').click();assert tab.locator('#viewer-title').inner_text().startswith('01.')
                tab.locator('#zoom').click();assert tab.locator('#zoom').get_attribute('aria-pressed')=='true'
                tab.locator('#zoom').click()
                tab.locator('dialog').evaluate("e=>{for(const [name,x] of [['touchstart',300],['touchend',100]]){const event=new Event(name);Object.defineProperty(event,'changedTouches',{value:[{clientX:x,clientY:100}]});e.dispatchEvent(event);}}")
                assert tab.locator('#viewer-title').inner_text().startswith('02.')
                tab.keyboard.press('Escape');assert not tab.locator('dialog').is_visible()
                assert tab.locator('.figure').first.evaluate('(e)=>e===document.activeElement')
                # Check every full image and source/download response locally.
                for m in data:assert context.request.get(base+m['full']).ok
                for target in ['src/analyze.py','pyproject.toml','Student_Bayes.code-workspace','reports/baseline_predictions.csv']:
                    assert context.request.get(base+target).ok
                tab.locator('#top-button').click();tab.wait_for_function('scrollY<5',timeout=5000)
                assert not errors,errors
                if args.capture:
                    folder=ROOT/'.review';folder.mkdir(exist_ok=True)
                    tab.screenshot(path=str(folder/f'layout-{width}-{theme}.png'))
                    tab.locator('#gallery').evaluate('(e)=>e.scrollIntoView({behavior:"instant",block:"start"})')
                    tab.wait_for_function("document.querySelector('nav a[href=\"#gallery\"]').getAttribute('aria-current')==='location'")
                    tab.screenshot(path=str(folder/f'gallery-{width}-{theme}.png'))
                out.append({'width':width,'theme':theme,'passed':True,'console_errors':errors,'checks':['readme-centered-header','numbered-contents','full-https-labels','background-selection','larger-nav-type','larger-back-to-top','final-author','overflow','grid','theme','svg-icons','mobile-menu','prediction-parity','presets','extrapolation','input-error','gallery-filter','empty-reset','filtered-wrap','keyboard','zoom','swipe-event','focus-return','downloads','back-to-top']})
                context.close()
        browser.close()
finally:server.shutdown();server.server_close()
(ROOT/'reports/site_checks.json').write_text(json.dumps(out,indent=2))
print('PASS: 24 figure pairs, parity, local links, 7 badges, 6 responsive/theme scenarios, prediction parity and controls')
