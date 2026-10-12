/* =========================================================================
 * app.js — ตัวแก้ไขผลการแปล (vanilla JS ไม่พึ่ง framework/CDN)
 *   1) โหลดรายการหน้า -> เลือกหน้า
 *   2) วาด overlay กรอบบอลลูนที่ลากย้าย/ปรับขนาดได้
 *   3) แก้คำแปล/ขนาดฟอนต์/การจัดชิด/สี แล้วบันทึก + เรนเดอร์ใหม่
 * ========================================================================= */

const state = {
  project: null,
  page: null,          // ข้อมูลหน้าที่กำลังแก้ (จาก /api/page/<idx>)
  selected: null,      // uid ของบอลลูนที่เลือก
  view: 'out',         // out | cleaned | orig
  dirty: false,
};

const $ = (id) => document.getElementById(id);
const imgEl = $('pageImage');
const overlayEl = $('overlay');

/* ------------------------- helpers ------------------------------ */
async function api(path, opts) {
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

let toastTimer = null;
function toast(msg, isErr = false) {
  const t = $('toast');
  t.textContent = msg;
  t.className = 'toast show' + (isErr ? ' err' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = 'toast'), 2600);
}

function curBubble() {
  return state.page?.bubbles.find((b) => b.uid === state.selected) || null;
}

/* ------------------------- project / pages ---------------------- */
async function loadProject() {
  state.project = await api('/api/project');
  const ul = $('pageList');
  ul.innerHTML = '';
  if (!state.project.pages.length) {
    ul.innerHTML = '<li class="muted">ไม่พบหน้า</li>';
    return;
  }
  for (const p of state.project.pages) {
    const li = document.createElement('li');
    li.className = p.out ? 'has-out' : '';
    li.dataset.index = p.index;
    li.innerHTML = `<span>หน้า ${p.index}</span><span class="dot"></span>`;
    li.onclick = () => selectPage(p.index);
    ul.appendChild(li);
  }
  selectPage(state.project.pages[0].index);
}

async function selectPage(index) {
  document.querySelectorAll('#pageList li').forEach((li) =>
    li.classList.toggle('active', Number(li.dataset.index) === index));
  state.selected = null;
  $('editPanel').classList.add('hidden');
  try {
    state.page = await api(`/api/page/${index}`);
  } catch (e) {
    toast(e.message, true);
    overlayEl.innerHTML = '';
    imgEl.removeAttribute('src');
    return;
  }
  state.dirty = false;
  refreshImage();
  renderOverlay();
  renderBubbleList();
  $('hint').textContent =
    `หน้า ${index} · ${state.page.width}×${state.page.height} · ${state.page.bubbles.length} บอลลูน`;
}

function refreshImage(force = false) {
  const m = state.page.media[state.view];
  if (!m) { imgEl.removeAttribute('src'); toast('ไม่มีภาพในมุมมองนี้', true); return; }
  imgEl.src = m + (force ? `?t=${Date.now()}` : '');
}

/* ------------------------- overlay ------------------------------ */
function overlayScale() {
  const rect = imgEl.getBoundingClientRect();
  return state.page ? rect.width / state.page.width : 1;
}

function autoFontSize(b) {
  // ประมาณขนาดฟอนต์ (ภาพจริงเรนเดอร์บนเซิร์ฟเวอร์) เพื่อเป็นแนวทางเท่านั้น
  const bh = b.bbox[3] - b.bbox[1];
  return Math.max(12, Math.min(64, Math.round(bh * 0.42)));
}

function renderOverlay() {
  overlayEl.innerHTML = '';
  if (!state.page) return;
  const show = $('showOverlay').checked;
  overlayEl.style.display = show ? 'block' : 'none';
  if (!show) return;

  const s = overlayScale();
  for (const b of state.page.bubbles) {
    const meta = b.meta || {};
    const box = meta.bbox || b.bbox;
    const off = meta.offset || [0, 0];
    const x = (box[0] + off[0]) * s;
    const y = (box[1] + off[1]) * s;
    const w = (box[2] - box[0]) * s;
    const h = (box[3] - box[1]) * s;

    const div = document.createElement('div');
    div.className = 'bub' + (b.uid === state.selected ? ' selected' : '');
    div.style.left = `${x}px`;
    div.style.top = `${y}px`;
    div.style.width = `${w}px`;
    div.style.height = `${h}px`;
    div.dataset.uid = b.uid;

    const fs = (meta.font_size || autoFontSize(b)) * s;
    const txt = document.createElement('div');
    txt.className = 'bub-text';
    txt.style.fontSize = `${fs}px`;
    if (meta.color) txt.style.color = `rgb(${meta.color.join(',')})`;
    if (meta.align === 'left') txt.style.justifyContent = 'flex-start';
    if (meta.align === 'right') txt.style.justifyContent = 'flex-end';
    txt.textContent = b.translation || b.source_text || '';
    div.appendChild(txt);

    const k = document.createElement('span');
    k.className = 'kind';
    k.textContent = b.kind;
    div.appendChild(k);

    const handle = document.createElement('div');
    handle.className = 'handle';
    div.appendChild(handle);

    overlayEl.appendChild(div);
  }
}

/* ------------------------- interactions ------------------------- */
let drag = null;

overlayEl.addEventListener('mousedown', (e) => {
  const handle = e.target.closest('.handle');
  const box = e.target.closest('.bub');
  if (!box) return;
  const uid = box.dataset.uid;
  selectBubble(uid);
  const b = curBubble();
  if (!b) return;

  const s = overlayScale();
  drag = {
    mode: handle ? 'resize' : 'move',
    uid, startX: e.clientX, startY: e.clientY,
    baseOffset: (b.meta?.offset || [0, 0]).slice(),
    baseBbox: (b.meta?.bbox || b.bbox).slice(),
    scale: s,
  };
  e.preventDefault();
});

window.addEventListener('mousemove', (e) => {
  if (!drag) return;
  const b = curBubble();
  if (!b) return;
  b.meta = b.meta || {};
  const dx = (e.clientX - drag.startX) / drag.scale;
  const dy = (e.clientY - drag.startY) / drag.scale;
  if (drag.mode === 'move') {
    b.meta.offset = [Math.round(drag.baseOffset[0] + dx), Math.round(drag.baseOffset[1] + dy)];
  } else {
    b.meta.bbox = [
      Math.round(drag.baseBbox[0]), Math.round(drag.baseBbox[1]),
      Math.round(drag.baseBbox[2] + dx), Math.round(drag.baseBbox[3] + dy),
    ];
  }
  state.dirty = true;
  renderOverlay();
});

window.addEventListener('mouseup', () => { if (drag) { drag = null; syncPanel(); } });
window.addEventListener('resize', renderOverlay);
imgEl.addEventListener('load', renderOverlay);

/* ------------------------- bubble list -------------------------- */
function renderBubbleList() {
  const list = $('bubbleList');
  list.innerHTML = '';
  for (const b of state.page.bubbles) {
    const d = document.createElement('div');
    d.className = 'bubble-item' + (b.uid === state.selected ? ' active' : '');
    d.innerHTML = `<div>${(b.translation || b.source_text || '(ว่าง)').slice(0, 40)}
      </div><span class="k">${b.kind}</span>`;
    d.onclick = () => selectBubble(b.uid);
    list.appendChild(d);
  }
}

function selectBubble(uid) {
  state.selected = uid;
  renderOverlay();
  renderBubbleList();
  syncPanel();
}

/* ------------------------- edit panel --------------------------- */
function syncPanel() {
  const b = curBubble();
  const panel = $('editPanel');
  if (!b) { panel.classList.add('hidden'); return; }
  panel.classList.remove('hidden');
  const meta = b.meta || {};
  $('selUid').textContent = b.uid;
  $('textInput').value = b.translation || '';
  const fs = meta.font_size || 0;
  $('fontSize').value = fs || autoFontSize(b);
  $('fontSizeVal').textContent = fs ? `${fs}px` : 'auto';
  document.querySelectorAll('input[name=align]').forEach((r) =>
    (r.checked = (meta.align || 'center') === r.value));
  $('colorOn').checked = !!meta.color;
  if (meta.color) $('colorInput').value = rgbToHex(meta.color);
}

const rgbToHex = (c) => '#' + c.map((v) => v.toString(16).padStart(2, '0')).join('');
const hexToRgb = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];

$('textInput').addEventListener('input', (e) => {
  const b = curBubble(); if (!b) return;
  b.translation = e.target.value; state.dirty = true;
  renderOverlay();
});
$('fontSize').addEventListener('input', (e) => {
  const b = curBubble(); if (!b) return;
  b.meta = b.meta || {}; b.meta.font_size = Number(e.target.value); state.dirty = true;
  $('fontSizeVal').textContent = `${e.target.value}px`;
  renderOverlay();
});
document.querySelectorAll('input[name=align]').forEach((r) =>
  r.addEventListener('change', (e) => {
    const b = curBubble(); if (!b) return;
    b.meta = b.meta || {}; b.meta.align = e.target.value; state.dirty = true;
    renderOverlay();
  }));
$('colorOn').addEventListener('change', (e) => {
  const b = curBubble(); if (!b) return;
  b.meta = b.meta || {};
  b.meta.color = e.target.checked ? hexToRgb($('colorInput').value) : null;
  state.dirty = true; renderOverlay();
});
$('colorInput').addEventListener('input', (e) => {
  if (!$('colorOn').checked) return;
  const b = curBubble(); if (!b) return;
  b.meta = b.meta || {}; b.meta.color = hexToRgb(e.target.value); state.dirty = true;
  renderOverlay();
});

function nudge(dx, dy) {
  const b = curBubble(); if (!b) return;
  b.meta = b.meta || {};
  const o = (b.meta.offset || [0, 0]).slice();
  b.meta.offset = [o[0] + dx, o[1] + dy]; state.dirty = true;
  renderOverlay();
}
$('nudgeL').onclick = (e) => nudge(-(e.shiftKey ? 10 : 2), 0);
$('nudgeR').onclick = (e) => nudge(e.shiftKey ? 10 : 2, 0);
$('nudgeU').onclick = (e) => nudge(0, -(e.shiftKey ? 10 : 2));
$('nudgeD').onclick = (e) => nudge(0, e.shiftKey ? 10 : 2);

$('btnReset').onclick = () => {
  const b = curBubble(); if (!b) return;
  b.meta = {}; state.dirty = true; renderOverlay(); syncPanel();
  toast('รีเซ็ตค่าที่แก้ของบอลลูนนี้แล้ว');
};

document.addEventListener('keydown', (e) => {
  if (!state.selected || ['TEXTAREA', 'INPUT'].includes(document.activeElement.tagName)) return;
  const step = e.shiftKey ? 10 : 2;
  if (e.key === 'ArrowLeft') nudge(-step, 0);
  else if (e.key === 'ArrowRight') nudge(step, 0);
  else if (e.key === 'ArrowUp') nudge(0, -step);
  else if (e.key === 'ArrowDown') nudge(0, step);
  else return;
  e.preventDefault();
});

/* ------------------------- save / render ------------------------ */
function collectEdits() {
  return state.page.bubbles.map((b) => ({
    uid: b.uid,
    translation: b.translation || '',
    meta: b.meta || {},
  }));
}

async function savePage(silent = false) {
  try {
    await api(`/api/page/${state.page.index}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bubbles: collectEdits() }),
    });
    state.dirty = false;
    if (!silent) toast('บันทึกแล้ว ✓');
    return true;
  } catch (e) { toast('บันทึกไม่สำเร็จ: ' + e.message, true); return false; }
}

async function renderPage() {
  if (!(await savePage(true))) return;
  toast('กำลังเรนเดอร์ใหม่...');
  try {
    const r = await api(`/api/page/${state.page.index}/render`, { method: 'POST' });
    state.page.media = r.media;
    if (!state.page.media[state.view]) state.view = 'out';
    $('viewSelect').value = state.view;
    refreshImage(true);
    toast('เรนเดอร์ใหม่แล้ว ✓');
  } catch (e) { toast('เรนเดอร์ไม่สำเร็จ: ' + e.message, true); }
}

$('btnSave').onclick = () => savePage(false);
$('btnRender').onclick = () => renderPage();
$('viewSelect').onchange = (e) => { state.view = e.target.value; refreshImage(true); };
$('showOverlay').onchange = renderOverlay;

/* ------------------------- boot -------------------------------- */
loadProject().catch((e) => toast('โหลดโปรเจกต์ไม่สำเร็จ: ' + e.message, true));
