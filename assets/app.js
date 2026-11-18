/* Student_Bayes: local, dependency-free portfolio interactions. */
'use strict';
const $ = selector => document.querySelector(selector);
const cards = [...document.querySelectorAll('.figure')];
const figures = JSON.parse($('#gallery-data').textContent);
const model = JSON.parse($('#model-data').textContent);
const dialog = $('dialog');
let visible = figures.map((_, i) => i), current = 0, opener = null;
const glyph = {
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/>',
  moon: '<path d="M20.8 13a8.8 8.8 0 1 1-9.8-9.8A7 7 0 0 0 20.8 13Z"/>',
  menu: '<path d="M3 6h18M3 12h18M3 18h18"/>',
  close: '<path d="m6 6 12 12M6 18 18 6"/>'
};
function svg(name) { return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${glyph[name]}</svg>`; }
function syncTheme() {
  const dark = document.documentElement.dataset.theme === 'dark';
  $('#theme').innerHTML = svg(dark ? 'sun' : 'moon');
  $('#theme').setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
}
syncTheme();
$('#theme').onclick = () => {
  document.documentElement.dataset.theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  try { localStorage.setItem('student-bayes-theme', document.documentElement.dataset.theme); } catch (_) { /* Preference storage is optional. */ }
  syncTheme();
};
function closeMenu() { $('#nav').classList.remove('open'); $('#menu').innerHTML = svg('menu'); $('#menu').setAttribute('aria-expanded', 'false'); $('#menu').setAttribute('aria-label', 'Open menu'); }
$('#menu').onclick = () => {
  const open = $('#nav').classList.toggle('open');
  $('#menu').innerHTML = svg(open ? 'close' : 'menu');
  $('#menu').setAttribute('aria-expanded', String(open));
  $('#menu').setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
};
document.querySelectorAll('nav a').forEach(a => a.onclick = closeMenu);
document.addEventListener('click', event => { if (!event.composedPath().includes(document.querySelector('header'))) closeMenu(); });
function filterFigures() {
  const query = $('#search').value.trim().toLowerCase(), category = $('#filter').value;
  visible = [];
  cards.forEach((card, i) => {
    card.hidden = !(category === 'All' || figures[i].category === category) || !`${figures[i].title} ${figures[i].caption}`.toLowerCase().includes(query);
    if (!card.hidden) visible.push(i);
  });
  $('#count').textContent = `${visible.length} figure${visible.length === 1 ? '' : 's'}`;
  $('#empty').hidden = visible.length !== 0;
}
$('#search').oninput = filterFigures; $('#filter').onchange = filterFigures;
$('#reset-filters').onclick = () => { $('#search').value = ''; $('#filter').value = 'All'; filterFigures(); $('#search').focus(); };
function showFigure(index) {
  current = index;
  const figure = figures[index];
  $('#large').src = figure.full; $('#large').alt = figure.title;
  $('#viewer-title').textContent = `${String(figure.id).padStart(2, '0')}. ${figure.title}`;
  $('#viewer-position').textContent = `${visible.indexOf(index) + 1} / ${visible.length} in current selection`;
  $('#caption').textContent = figure.caption; $('#original').href = figure.full;
  $('#large').classList.remove('zoomed'); $('#zoom').setAttribute('aria-pressed', 'false');
  $('#image-error').hidden = true; $('.zoom-area').scrollTo(0, 0);
}
$('#large').onerror = () => { $('#image-error').hidden = false; };
cards.forEach((card, index) => card.onclick = () => { opener = card; showFigure(index); dialog.showModal(); });
function move(step) { if (visible.length) showFigure(visible[(visible.indexOf(current) + step + visible.length) % visible.length]); }
$('#previous').onclick = () => move(-1); $('#next').onclick = () => move(1);
$('#close').onclick = () => dialog.close(); dialog.addEventListener('close', () => opener?.focus());
$('#zoom').onclick = () => $('#zoom').setAttribute('aria-pressed', String($('#large').classList.toggle('zoomed')));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') closeMenu();
  if (!dialog.open) return;
  if (event.key === 'ArrowRight') { event.preventDefault(); move(1); }
  if (event.key === 'ArrowLeft') { event.preventDefault(); move(-1); }
});
let touchStart = null;
dialog.addEventListener('touchstart', event => { const t=event.changedTouches[0]; touchStart = {x:t.clientX,y:t.clientY}; }, {passive:true});
dialog.addEventListener('touchend', event => {
  if (!touchStart || $('#large').classList.contains('zoomed')) return;
  const touch=event.changedTouches[0], dx=touch.clientX-touchStart.x, dy=touch.clientY-touchStart.y;
  if (Math.abs(dx)>60 && Math.abs(dx)>Math.abs(dy)) move(dx<0?1:-1);
  touchStart=null;
}, {passive:true});
$('#top-button').onclick = () => window.scrollTo({top:0,behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'instant':'smooth'});
window.addEventListener('scroll', () => { $('#top-button').hidden=window.scrollY<500; }, {passive:true});
const navLinks=[...document.querySelectorAll('nav a')];
const observed=[...document.querySelectorAll('main section[id]')].filter(section=>navLinks.some(a=>a.hash==='#'+section.id));
let navigationScheduled=false;
function updateNavigation(){
  navigationScheduled=false;
  const edge=document.querySelector('header').getBoundingClientRect().height+100;
  let active=observed[0];
  for(const section of observed){if(section.getBoundingClientRect().top<=edge)active=section;}
  navLinks.forEach(a=>{const selected=a.hash==='#'+active.id;a.classList.toggle('active',selected);if(selected)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});
}
window.addEventListener('scroll',()=>{if(!navigationScheduled){navigationScheduled=true;requestAnimationFrame(updateNavigation);}},{passive:true});
window.addEventListener('resize',updateNavigation);updateNavigation();
/* The browser and CLI share the exact exported training parameters. */
function probabilities(values) {
  const scaled=values.map((v,i)=>(v-model.mean[i])/model.scale[i]);
  const scores=model.coefficients.map((row,k)=>row.reduce((sum,c,i)=>sum+c*scaled[i],model.intercept[k]));
  const max=Math.max(...scores), exp=scores.map(v=>Math.exp(v-max)), total=exp.reduce((a,b)=>a+b,0);
  return exp.map(v=>v/total);
}
function predict() {
  const values=[0,1,2].map(i=>Number($(`#habit-${i}`).value));
  const invalid=values.some((v,i)=>!$(`#habit-${i}`).value.trim() || !Number.isFinite(v) || v<0 || v>100);
  $('#predict-error').hidden=!invalid;
  $('#prediction').hidden=invalid;
  if(invalid){$('#predict-error').textContent='Enter three numeric hour values between 0 and 100.';return;}
  const probs=probabilities(values), best=probs.indexOf(Math.max(...probs));
  $('#predicted-class').textContent=`Highest model probability: ${model.classes[best]}`;
  $('#probability-bars').replaceChildren();
  model.classes.forEach((name,i)=>{
    const row=document.createElement('div');row.className='prob-row';
    const label=document.createElement('span');label.textContent=name;
    const track=document.createElement('div');track.className='prob-track';
    const fill=document.createElement('div');fill.className='prob-fill';fill.style.width=`${probs[i]*100}%`;track.append(fill);
    const value=document.createElement('span');value.className='prob-value';value.textContent=`${(probs[i]*100).toFixed(1)}%`;value.dataset.probability=String(probs[i]);
    row.append(label,track,value);$('#probability-bars').append(row);
  });
  const outside=values.some((v,i)=>v<model.observed_min[i] || v>model.observed_max[i]);
  $('#extrapolation').hidden=!outside;
  $('#extrapolation').textContent='Extrapolation: one or more inputs fall outside observed training ranges. Interpret these probabilities with extra caution.';
}
$('#predict-form').onsubmit=event=>{event.preventDefault();predict();};
document.querySelectorAll('.preset').forEach(button=>button.onclick=()=>{JSON.parse(button.dataset.values).forEach((v,i)=>{$(`#habit-${i}`).value=v;});predict();});
predict();
document.querySelectorAll('.copy-code').forEach(button=>button.onclick=async()=>{
  try{await navigator.clipboard.writeText(button.parentElement.querySelector('code').textContent);button.textContent='Copied';setTimeout(()=>button.textContent='Copy',1800);}
  catch(_){button.textContent='Select and copy';}
});
/* Background selection changes only the gallery mat; evidence images stay identical. */
const figureBackground = document.querySelector('#figure-background');
function applyFigureBackground(value) {
  document.querySelector('.gallery').dataset.background = value;
  figureBackground.value = value;
}
try {
  const savedBackground = localStorage.getItem('student-bayes-figure-background');
  if (['slate', 'warm', 'graphite'].includes(savedBackground)) applyFigureBackground(savedBackground);
} catch (_) { /* Local preferences are optional. */ }
figureBackground.addEventListener('change', () => {
  applyFigureBackground(figureBackground.value);
  try { localStorage.setItem('student-bayes-figure-background', figureBackground.value); } catch (_) {}
});
