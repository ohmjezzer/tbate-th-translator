/* =========================================================================
 * app.js — ตัวแก้ไขผลการแปล (vanilla JS ไม่พึ่ง framework/CDN)
 *   1) โหลดรายการหน้า -> เลือกหน้า
 *   2) โหมด เลือก/ย้าย  : ลากย้าย/ปรับกรอบบอลลูน + แก้คำแปล/ฟอนต์/สี
 *   3) โหมด แปรงลบ      : วาดทับพื้นที่ -> ให้ AI inpaint ลบพื้นหลัง
 *   4) โหมด เพิ่มบอลลูน : ลากวาดกรอบใหม่ ; ปุ่ม Delete = ลบบอลลูน
 * ========================================================================= */

const state = {
  project: null,
  page: null,
  pageIndex: null,
  selected: null,
  view: 'out',
  dirty: false,
  tool: 'select',           // select | brush | add
  strokes: [],              // [{points:[[x,y]...], radius}]
  drawing: null,            // stroke ที่กำลังวาด
  adding: null,             // {x0,y0,x1,y1} กำลังวาดกรอบใหม่
  brushSize: 24,
  busy: false,
};

const $ = (id) => document.getElementById(id);
const imgEl = $('pageImage');
const overlayEl = $('overlay');
const canvas = $('brushCanvas');
const ctx = canvas.getContext('2d');

/* ------------------------- helpers ------------------------------ */
async function api(path, opts) {
  const res = await fetch(path, opts);
  let data = {};
  try { data = await res.json(); } catch (e) { /* ไม่ใช่ JSON */ }
  if (!res.ok) {
    const err = new Error(data.error || `HTTP ${res.status}`);
    err.body = data; err.status = res.status;
    throw err;
  }
  return data;
}

let toastTimer = null;
function toast(msg, isErr = false) {
  const t = $('toast');
  t.textContent = msg;
  t.className = 'toast show' + (isErr ? ' err' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = 'toast'), 3200);
}

const curBubble = () => state.page?.bubbles.find((b) => b.uid === state.selected) || null;
const scale = () => {
  const r = imgEl.getBoundingClientRect();
  return state.page ? r.width / state.page.width : 1;
};
const rgbToHex = (c) => '#' + c.map((v) => Math.round(v).toString(16).padStart(2, '0')).join('');
const hexToRgb = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];

/* ------------------------- project / pages ---------------------- */
async function loadProject() {
  state.project = await api('/api/project');
  const ul = $('pageList');
  ul.innerHTML = '';
  if (!state.project.pages.length) { ul.innerHTML = '<li class="muted">ไม่พบหน้า</li>'; return; }
  for (const p of state.project.pages) {
    const li = document.createElement('li');
    li.className = (p.out ? 'has-out ' : '') + (p.ready ? '' : 'not-ready');
    li.dataset.index = p.index;
    li.title = p.ready ? 'พร้อมแก้ไข' : 'ยังไม่ได้ประมวลผล';
    li.innerHTML = `<span>หน้า ${p.index}</span>` +
      (p.ready ? '<span class="dot"></span>'
               : '<span class="badge">ยังไม่ประมวลผล</span><span class="dot"></span>');
    li.onclick = () => selectPage(p.index);
    ul.appendChild(li);
  }
  const first = state.project.pages.find((p) => p.ready) || state.project.pages[0];
  selectPage(first.index);
}

async function selectPage(index) {
  document.querySelectorAll('#pageList li').forEach((li) =>
    li.classList.toggle('active', Number(li.dataset.index) === index));
  state.selected = null;
  state.pageIndex = index;
  $('editPanel').classList.add('hidden');
  $('btnDeleteBubble').disabled = true;
  clearBrush();
  try {
    state.page = await api(`/api/page/${index}`);
  } catch (e) {
    state.page = null;
    hideOverlayAndImage();
    showEmptyState(index, e.body || { error: e.message });
    return;
  }
  hideEmptyState();
  state.dirty = false;
  imgEl.onload = () => { renderOverlay(); resizeCanvas(); drawBrush(); };
  refreshImage();
  renderOverlay();
  renderBubbleList();
  syncPanel();
  $('hint').textContent =
    `หน้า ${index} · ${state.page.width}×${state.page.height} · ${state.page.bubbles.length} บอลลูน`;
}

function hideOverlayAndImage() {
  overlayEl.innerHTML = ''; imgEl.removeAttribute('src');
  canvas.width = 0; canvas.height = 0; $('hint').textContent = '';
}

function showEmptyState(index, info) {
  const es = $('emptyState');
  es.classList.remove('hidden');
  $('esTitle').textContent = info.needs_run ? 'หน้านี้ยังไม่ได้ประมวลผล' : 'เปิดหน้านี้ไม่ได้';
  $('esMsg').textContent = info.error || '';
  $('esCmd').textContent = info.command || `python -m scanlation.cli run ${index}`;
  $('esHint').textContent = info.hint || '';
  $('btnPrepare').dataset.index = index;
  $('btnPrepare').disabled = false;
  $('btnPrepare').textContent = '▶ ประมวลผลหน้านี้';
}
const hideEmptyState = () => $('emptyState').classList.add('hidden');

function refreshImage(force = false) {
  const m = state.page.media[state.view];
  if (!m) { imgEl.removeAttribute('src'); toast('ไม่มีภาพในมุมมองนี้', true); return; }
  imgEl.src = m + (force ? `?t=${Date.now()}` : '');
}

/* ------------------------- overlay (กรอบบอลลูน) ----------------- */
function autoFontSize(b) {
  const bh = b.bbox[3] - b.bbox[1];
  return Math.max(12, Math.min(64, Math.round(bh * 0.42)));
}

function boxOf(b) {
  const meta = b.meta || {};
  const box = meta.bbox || b.bbox;
  const off = meta.offset || [0, 0];
  return { x: box[0] + off[0], y: box[1] + off[1], w: box[2] - box[0], h: box[3] - box[1] };
}

function renderOverlay() {
  overlayEl.innerHTML = '';
  if (!state.page) return;
  const show = $('showOverlay').checked;
  overlayEl.style.display = show ? 'block' : 'none';
  if (!show) return;
  const s = scale();
  for (const b of state.page.bubbles) {
    const { x, y, w, h } = boxOf(b);
    const meta = b.meta || {};
    const div = document.createElement('div');
    div.className = 'bub' + (b.uid === state.selected ? ' selected' : '');
    div.style.left = `${x * s}px`; div.style.top = `${y * s}px`;
    div.style.width = `${w * s}px`; div.style.height = `${h * s}px`;
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

    const k = document.createElement('span'); k.className = 'kind'; k.textContent = b.kind;
    div.appendChild(k);
    const handle = document.createElement('div'); handle.className = 'handle'; div.appendChild(handle);
    const del = document.createElement('div'); del.className = 'del-badge'; del.textContent = '✕';
    del.title = 'ลบบอลลูนนี้';
    del.onclick = (ev) => { ev.stopPropagation(); deleteBubble(b.uid); };
    div.appendChild(del);

    overlayEl.appendChild(div);
  }
}

/* ------------------------- interactions ------------------------- */
imgEl.addEventListener('load', () => { renderOverlay(); resizeCanvas(); drawBrush(); });
window.addEventListener('resize', () => { renderOverlay(); resizeCanvas(); drawBrush(); });

/* ---- โหมด 1: เลือก/ย้าย (บน overlay) ---- */
let drag = null;
overlayEl.addEventListener('mousedown', (e) => {
  if (state.tool !== 'select') return;
  const handle = e.target.closest('.handle');
  const box = e.target.closest('.bub');
  if (!box) return;
  selectBubble(box.dataset.uid);
  const b = curBubble(); if (!b) return;
  drag = {
    mode: handle ? 'resize' : 'move', uid: b.uid,
    startX: e.clientX, startY: e.clientY, scale: scale(),
    baseOffset: (b.meta?.offset || [0, 0]).slice(),
    baseBbox: (b.meta?.bbox || b.bbox).slice(),
  };
  e.preventDefault();
});
window.addEventListener('mousemove', (e) => {
  if (!drag) return;
  const b = curBubble(); if (!b) return;
  b.meta = b.meta || {};
  const dx = (e.clientX - drag.startX) / drag.scale;
  const dy = (e.clientY - drag.startY) / drag.scale;
  if (drag.mode === 'move') {
    b.meta.offset = [Math.round(drag.baseOffset[0] + dx), Math.round(drag.baseOffset[1] + dy)];
  } else {
    b.meta.bbox = [Math.round(drag.baseBbox[0]), Math.round(drag.baseBbox[1]),
                   Math.round(drag.baseBbox[2] + dx), Math.round(drag.baseBbox[3] + dy)];
  }
  state.dirty = true; renderOverlay();
});
window.addEventListener('mouseup', () => { if (drag) { drag = null; syncPanel(); } });

/* ---- โหมด 2: แปรงลบ ---- */
function resizeCanvas() {
  const r = imgEl.getBoundingClientRect();
  if (!r.width) return;
  canvas.width = Math.round(r.width); canvas.height = Math.round(r.height);
  canvas.style.width = `${r.width}px`; canvas.style.height = `${r.height}px`;
}

function drawBrush() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const s = scale();
  ctx.fillStyle = 'rgba(255,60,90,0.45)';
  ctx.strokeStyle = 'rgba(255,60,90,0.45)';
  const all = state.drawing ? state.strokes.concat([state.drawing]) : state.strokes;
  for (const st of all) {
    const pts = st.points || [];
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p[0] * s, p[1] * s, Math.max(1, st.radius * s), 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

canvas.addEventListener('mousedown', (e) => {
  if (state.tool !== 'brush') return;
  const r = canvas.getBoundingClientRect();
  const s = scale();
  const x = (e.clientX - r.left) / s, y = (e.clientY - r.top) / s;
  state.drawing = { points: [[x, y]], radius: state.brushSize };
  drawBrush();
  e.preventDefault();
});
canvas.addEventListener('mousemove', (e) => {
  if (!state.drawing) return;
  const r = canvas.getBoundingClientRect();
  const s = scale();
  state.drawing.points.push([(e.clientX - r.left) / s, (e.clientY - r.top) / s]);
  drawBrush();
});
window.addEventListener('mouseup', () => {
  if (state.drawing) {
    state.strokes.push(state.drawing);
    state.drawing = null;
    drawBrush();
  }
});

function clearBrush() { state.strokes = []; state.drawing = null; drawBrush(); }

/* ---- โหมด 3: เพิ่มบอลลูน ---- */
let addRect = null;
canvas.addEventListener('mousedown', (e) => {
  if (state.tool !== 'add') return;
  const r = canvas.getBoundingClientRect();
  const s = scale();
  addRect = { x0: (e.clientX - r.left) / s, y0: (e.clientY - r.top) / s,
              x1: (e.clientX - r.left) / s, y1: (e.clientY - r.top) / s };
  e.preventDefault();
});
canvas.addEventListener('mousemove', (e) => {
  if (!addRect) return;
  const r = canvas.getBoundingClientRect(); const s = scale();
  addRect.x1 = (e.clientX - r.left) / s; addRect.y1 = (e.clientY - r.top) / s;
  drawAddRect();
});
window.addEventListener('mouseup', async () => {
  if (!addRect) return;
  const a = addRect; addRect = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const x0 = Math.min(a.x0, a.x1), y0 = Math.min(a.y0, a.y1);
  const x1 = Math.max(a.x0, a.x1), y1 = Math.max(a.y0, a.y1);
  if (x1 - x0 < 8 || y1 - y0 < 8) { toast('กรอบเล็กเกินไป', true); return; }
  await addBubble([Math.round(x0), Math.round(y0), Math.round(x1), Math.round(y1)]);
});

function drawAddRect() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!addRect) return;
  const s = scale();
  const x = Math.min(addRect.x0, addRect.x1) * s, y = Math.min(addRect.y0, addRect.y1) * s;
  const w = Math.abs(addRect.x1 - addRect.x0) * s, h = Math.abs(addRect.y1 - addRect.y0) * s;
  ctx.fillStyle = 'rgba(46,204,113,0.18)'; ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = '#2ecc71'; ctx.lineWidth = 2; ctx.strokeRect(x, y, w, h);
}

/* ------------------------- bubble list / panel ------------------ */
function renderBubbleList() {
  const list = $('bubbleList');
  list.innerHTML = '';
  for (const b of state.page.bubbles) {
    const d = document.createElement('div');
    d.className = 'bubble-item' + (b.uid === state.selected ? ' active' : '');
    d.innerHTML = `<div>${(b.translation || b.source_text || '(ว่าง)').slice(0, 40)}</div>
      <span class="k">${b.kind}</span>`;
    d.onclick = () => selectBubble(b.uid);
    list.appendChild(d);
  }
}

function selectBubble(uid) {
  state.selected = uid;
  renderOverlay(); renderBubbleList(); syncPanel();
}

function syncPanel() {
  const b = curBubble();
  const panel = $('editPanel');
  $('btnDeleteBubble').disabled = !b;
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
  if (b.kind) $('kindSel').value = b.kind;
}

$('textInput').addEventListener('input', (e) => {
  const b = curBubble(); if (!b) return;
  b.translation = e.target.value; state.dirty = true; renderOverlay();
});
$('fontSize').addEventListener('input', (e) => {
  const b = curBubble(); if (!b) return;
  b.meta = b.meta || {}; b.meta.font_size = Number(e.target.value); state.dirty = true;
  $('fontSizeVal').textContent = `${e.target.value}px`; renderOverlay();
});
document.querySelectorAll('input[name=align]').forEach((r) =>
  r.addEventListener('change', (e) => {
    const b = curBubble(); if (!b) return;
    b.meta = b.meta || {}; b.meta.align = e.target.value; state.dirty = true; renderOverlay();
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
  b.meta = b.meta || {}; b.meta.color = hexToRgb(e.target.value); state.dirty = true; renderOverlay();
});
$('kindSel').addEventListener('change', (e) => {
  const b = curBubble(); if (!b) return;
  b.kind = e.target.value; state.dirty = true; renderOverlay(); renderBubbleList();
});

function nudge(dx, dy) {
  const b = curBubble(); if (!b) return;
  b.meta = b.meta || {};
  const o = (b.meta.offset || [0, 0]).slice();
  b.meta.offset = [o[0] + dx, o[1] + dy]; state.dirty = true; renderOverlay();
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
  if (['TEXTAREA', 'INPUT', 'SELECT'].includes(document.activeElement.tagName)) return;
  if ((e.key === 'Delete' || e.key === 'Backspace') && state.selected) {
    e.preventDefault(); deleteBubble(state.selected); return;
  }
  if (!state.selected) return;
  const step = e.shiftKey ? 10 : 2;
  const map = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
  if (map[e.key]) { e.preventDefault(); nudge(...map[e.key]); }
});

/* ------------------------- tool switching ----------------------- */
function setTool(tool) {
  state.tool = tool;
  document.querySelectorAll('.tool').forEach((b) =>
    b.classList.toggle('active', b.dataset.tool === tool));
  $('brushControls').classList.toggle('hidden', tool !== 'brush');
  $('addControls').classList.toggle('hidden', tool !== 'add');
  const inner = $('stageInner');
  inner.classList.toggle('brushing', tool === 'brush');
  inner.classList.toggle('adding', tool === 'add');
  canvas.style.cursor = tool === 'select' ? 'default' : 'crosshair';
  overlayEl.style.pointerEvents = tool === 'select' ? 'auto' : 'none';
  if (tool !== 'brush') { /* เก็บ strokes ไว้ แต่ซ่อนสี */ }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (tool === 'brush') drawBrush();
}
$('toolSelect').onclick = () => setTool('select');
$('toolBrush').onclick = () => setTool('brush');
$('toolAdd').onclick = () => setTool('add');
$('brushSize').addEventListener('input', (e) => {
  state.brushSize = Number(e.target.value); $('brushSizeVal').textContent = e.target.value;
});
$('btnClearBrush').onclick = () => { clearBrush(); toast('ล้างที่แปรงแล้ว'); };

/* ------------------------- save / render / inpaint -------------- */
function collectEdits() {
  return state.page.bubbles.map((b) => ({ uid: b.uid, translation: b.translation || '', meta: b.meta || {} }));
}

async function savePage(silent = false) {
  try {
    await api(`/api/page/${state.page.index}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
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
    if (!state.page.media[state.view]) { state.view = 'out'; $('viewSelect').value = 'out'; }
    refreshImage(true);
    toast('เรนเดอร์ใหม่แล้ว ✓');
  } catch (e) { toast('เรนเดอร์ไม่สำเร็จ: ' + e.message, true); }
}

async function addBubble(bbox) {
  try {
    const r = await api(`/api/page/${state.page.index}/bubbles`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ op: 'add', bbox, kind: $('newKind').value }),
    });
    await selectPage(state.page.index);
    selectBubble(r.uid);
    setTool('select');
    toast('เพิ่มบอลลูนแล้ว ✓ (พิมพ์คำแปลได้เลย)');
  } catch (e) { toast('เพิ่มบอลลูนไม่สำเร็จ: ' + e.message, true); }
}

async function deleteBubble(uid) {
  if (!uid) return;
  try {
    await api(`/api/page/${state.page.index}/bubbles`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ op: 'delete', uid }),
    });
    await selectPage(state.page.index);
    toast('ลบบอลลูนแล้ว ✓');
  } catch (e) { toast('ลบไม่สำเร็จ: ' + e.message, true); }
}

async function applyInpaint() {
  if (!state.strokes.length) { toast('ยังไม่ได้แปรงพื้นที่', true); return; }
  state.busy = true;
  $('btnApplyInpaint').disabled = true;
  $('btnApplyInpaint').textContent = '⏳ กำลังลบ (AI)...';
  try {
    await api(`/api/page/${state.page.index}/inpaint`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ op: 'inpaint', strokes: state.strokes, grow: 4 }),
    });
    clearBrush();
    refreshImage(true);
    toast('ลบพื้นที่เรียบร้อย ✓');
  } catch (e) {
    toast('ลบไม่สำเร็จ: ' + (e.body?.error || e.message), true);
  } finally {
    state.busy = false;
    $('btnApplyInpaint').disabled = false;
    $('btnApplyInpaint').textContent = '🪄 ลบพื้นที่ที่แปรง (AI)';
  }
}

async function undoInpaint() {
  try {
    await api(`/api/page/${state.page.index}/inpaint`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ op: 'undo' }),
    });
    refreshImage(true);
    toast('ย้อนการลบแล้ว ✓');
  } catch (e) { toast('undo ไม่สำเร็จ: ' + (e.body?.error || e.message), true); }
}

$('btnSave').onclick = () => savePage(false);
$('btnRender').onclick = () => renderPage();
$('viewSelect').onchange = (e) => { state.view = e.target.value; refreshImage(true); };
$('showOverlay').onchange = renderOverlay;
$('btnApplyInpaint').onclick = applyInpaint;
$('btnUndoInpaint').onclick = undoInpaint;
$('btnDeleteBubble').onclick = () => deleteBubble(state.selected);
$('btnPrepare').onclick = () => preparePage(Number($('btnPrepare').dataset.index));

async function preparePage(index) {
  const btn = $('btnPrepare');
  btn.disabled = true; const original = btn.textContent;
  btn.textContent = '⏳ กำลังประมวลผล... (อาจใช้เวลาหลายวินาที)';
  try {
    await api(`/api/page/${index}/prepare`, { method: 'POST' });
    toast('ประมวลผลหน้านี้เสร็จแล้ว ✓');
    await loadProject();
    await selectPage(index);
  } catch (e) {
    const info = e.body || {};
    $('esHint').textContent = (info.error || e.message) + (info.hint ? ' — ' + info.hint : '');
    toast('ประมวลผลไม่สำเร็จ: ' + (info.error || e.message), true);
    btn.disabled = false; btn.textContent = original;
  }
}

/* ------------------------- boot -------------------------------- */
setTool('select');
loadProject().catch((e) => toast('โหลดโปรเจกต์ไม่สำเร็จ: ' + e.message, true));
