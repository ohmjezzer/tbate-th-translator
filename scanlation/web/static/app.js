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
  brushErase: false,        // โหมดแปรง: true = ลบเส้นที่แปรงไว้
  hover: null,              // พิกัดเมาส์บนภาพ (ไว้โชว์วงกลมขนาดแปรง)
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

/* -------------------------------------------------------------------------
 * การแปลงพิกัด (สำคัญมาก — ต้นตอ "แปรงลบ/วาดกรอบแล้วได้ผิดตำแหน่ง")
 * เดิมใช้กรอบของ <canvas> เป็นตัวอ้างอิง ซึ่งอาจไม่ตรงกับรูปที่แสดงจริง
 * (เช่น รูปถูกย่อ/สลับ view out/cleaned/orig ขนาดต่างกัน)
 * ตอนนี้ยึด "กรอบของ <img> + ขนาดจริง (natural)" เป็นมาตรฐานเดียว
 *   - toImageXY(ev)  : พิกัดเมาส์ -> พิกัดพิกเซลของภาพ (ใช้ป้อน stroke/กรอบ)
 *   - scaleXY()      : อัตราส่วนแสดงผล/ขนาดจริง (แกน x, y แยกกัน)
 * ----------------------------------------------------------------------- */
function imgNatural() {
  const nw = imgEl.naturalWidth || state.page?.width || 1;
  const nh = imgEl.naturalHeight || state.page?.height || 1;
  return { nw, nh };
}
function scaleXY() {
  const r = imgEl.getBoundingClientRect();
  const { nw, nh } = imgNatural();
  return { sx: r.width ? r.width / nw : 1, sy: r.height ? r.height / nh : 1 };
}
const scale = () => scaleXY().sx;
function toImageXY(ev) {
  const r = imgEl.getBoundingClientRect();
  const { sx, sy } = scaleXY();
  const { nw, nh } = imgNatural();
  const x = Math.max(0, Math.min(nw, (ev.clientX - r.left) / sx));
  const y = Math.max(0, Math.min(nh, (ev.clientY - r.top) / sy));
  return { x, y };
}
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
  loadMetrics(index);
  showOutputInfo();
  updateStatus();
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
  const { sx, sy } = scaleXY();
  const all = state.drawing ? state.strokes.concat([state.drawing]) : state.strokes;
  for (const st of all) {
    const pts = st.points || [];
    const rx = Math.max(1, st.radius * sx), ry = Math.max(1, st.radius * sy);
    const color = 'rgba(255,60,90,0.45)';
    // จุด
    ctx.fillStyle = color;
    for (const p of pts) {
      ctx.beginPath(); ctx.ellipse(p[0] * sx, p[1] * sy, rx, ry, 0, 0, Math.PI * 2); ctx.fill();
    }
    // เส้นเชื่อมให้ดูต่อเนื่อง (ตรงกับที่ backend วาดจริง)
    if (pts.length >= 2) {
      ctx.strokeStyle = color; ctx.lineWidth = Math.max(2, st.radius * 2 * sx);
      ctx.lineCap = ctx.lineJoin = 'round';
      ctx.beginPath(); ctx.moveTo(pts[0][0] * sx, pts[0][1] * sy);
      for (const p of pts.slice(1)) ctx.lineTo(p[0] * sx, p[1] * sy);
      ctx.stroke();
    }
  }
  moveBrushCursor(state.hover);
}

/* วงกลมแสดงขนาดแปรง (ให้เห็นก่อนวาดว่ากว้างแค่ไหน) */
function moveBrushCursor(imgPt) {
  const cur = $('brushCursor');
  if (!cur) return;
  if (!imgPt || (state.tool !== 'brush' && state.tool !== 'add')) {
    cur.style.display = 'none'; return;
  }
  const s = scale();
  const rr = Math.max(3, state.brushSize * s);
  cur.style.display = 'block';
  cur.style.width = cur.style.height = `${rr * 2}px`;
  cur.style.left = `${imgPt[0] * s - rr}px`;
  cur.style.top = `${imgPt[1] * s - rr}px`;
}

canvas.addEventListener('mousedown', (e) => {
  if (state.tool !== 'brush') return;
  const { x, y } = toImageXY(e);
  if (state.brushErase) { eraseStrokeAt(x, y); return; }
  state.drawing = { points: [[x, y]], radius: state.brushSize };
  drawBrush(); e.preventDefault();
});
canvas.addEventListener('mousemove', (e) => {
  const { x, y } = toImageXY(e);
  state.hover = [x, y];
  if (state.tool === 'brush' || state.tool === 'add') moveBrushCursor(state.hover);
  if (!state.drawing) return;
  state.drawing.points.push([x, y]);
  drawBrush();
});
canvas.addEventListener('mouseleave', () => { state.hover = null; moveBrushCursor(null); });
window.addEventListener('mouseup', () => {
  if (state.drawing) {
    state.strokes.push(state.drawing);
    state.drawing = null;
    drawBrush();
    updateStatus();
  }
});

/* ลบเส้นที่แปรงไว้ (คลิกทับ) — ใช้เมื่อวาดผิดจุด */
function eraseStrokeAt(x, y) {
  const idx = state.strokes.findIndex((st) =>
    (st.points || []).some((pt) => Math.hypot(pt[0] - x, pt[1] - y) <= (st.radius + state.brushSize)));
  if (idx >= 0) {
    state.strokes.splice(idx, 1);
    drawBrush(); updateStatus();
    toast('ลบเส้นที่แปรงแล้ว');
  } else {
    toast('ไม่พบเส้นที่ตำแหน่งนี้ — คลิกให้ตรงเส้นสีแดง', true);
  }
}

function undoStroke() {
  if (!state.strokes.length) { toast('ไม่มีเส้นให้ลบ', true); return; }
  state.strokes.pop(); drawBrush(); updateStatus();
  toast('ลบเส้นล่าสุดแล้ว');
}

function toggleBrushErase() {
  state.brushErase = !state.brushErase;
  $('brushModeErase').classList.toggle('toggle-on', state.brushErase);
  $('brushModeErase').textContent = state.brushErase ? '🧽 กำลังลบเส้น (คลิกทับ)' : '🧽 ลบเส้นที่แปรง';
  canvas.style.cursor = state.brushErase ? 'not-allowed' : 'crosshair';
  updateStatus();
}

function clearBrush() { state.strokes = []; state.drawing = null; drawBrush(); updateStatus(); }

/* ---- โหมด 3: เพิ่มบอลลูน ---- */
let addRect = null;
canvas.addEventListener('mousedown', (e) => {
  if (state.tool !== 'add') return;
  const { x, y } = toImageXY(e);
  addRect = { x0: x, y0: y, x1: x, y1: y };
  e.preventDefault();
});
canvas.addEventListener('mousemove', (e) => {
  if (!addRect) return;
  const { x, y } = toImageXY(e);
  addRect.x1 = x; addRect.y1 = y;
  drawAddRect();
});
window.addEventListener('mouseup', async () => {
  if (!addRect) return;
  const a = addRect; addRect = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const x0 = Math.round(Math.min(a.x0, a.x1)), y0 = Math.round(Math.min(a.y0, a.y1));
  const x1 = Math.round(Math.max(a.x0, a.x1)), y1 = Math.round(Math.max(a.y0, a.y1));
  if (x1 - x0 < 12 || y1 - y0 < 12) {
    toast('กรอบเล็กเกินไป — ลากให้ใหญ่กว่า 12×12 px', true);
    updateStatus();
    return;
  }
  await addBubble([x0, y0, x1, y1], $('newErase') ? $('newErase').checked : false);
});

function drawAddRect() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!addRect) return;
  const { sx, sy } = scaleXY();
  const x = Math.min(addRect.x0, addRect.x1) * sx, y = Math.min(addRect.y0, addRect.y1) * sy;
  const w = Math.abs(addRect.x1 - addRect.x0) * sx, h = Math.abs(addRect.y1 - addRect.y0) * sy;
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
  updateStatus();
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
  overlayEl.style.pointerEvents = tool === 'select' ? 'auto' : 'none';

  // ออกจากโหมดแปรง -> ปิดโหมดลบเส้นด้วย (กันสับสน)
  if (tool !== 'brush' && state.brushErase) {
    state.brushErase = false;
    $('brushModeErase').classList.remove('toggle-on');
    $('brushModeErase').textContent = '🧽 ลบเส้นที่แปรง';
  }
  canvas.style.cursor = (tool === 'brush' && state.brushErase) ? 'not-allowed' : 'crosshair';
  if (tool === 'select') canvas.style.cursor = 'default';

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (tool === 'brush') drawBrush();
  moveBrushCursor(null);
  updateStatus();
}

/* แถบสถานะ: บอกทีละขั้นว่ากำลังทำอะไร/ต้องทำอะไรต่อ */
function updateStatus() {
  const ico = $('statusIco'), txt = $('statusText');
  if (!ico || !txt) return;
  const b = curBubble();
  let icon = '💡', msg = '';
  if (state.tool === 'brush') {
    icon = '🖌';
    msg = state.brushErase
      ? 'โหมดลบเส้น: คลิกทับ "เส้นสีแดง" ที่แปรงไว้เพื่อลบทิ้ง'
      : `วาดทับข้อความ/พื้นหลังที่อยากลบ (จะเห็นสีแดง) แล้วกด “🪄 ลบพื้นที่ที่แปรง (AI)” · ขนาดแปรง ${state.brushSize}px · แปรงไว้ ${state.strokes.length} เส้น`;
  } else if (state.tool === 'add') {
    icon = '➕';
    msg = 'ลากวาดกรอบครอบข้อความที่ระบบจับไม่เจอ → ระบบจะสร้างบอลลูนใหม่ให้ (ติ๊ก “ลบข้อความเดิมในกรอบ” ได้)';
  } else {
    icon = '✋';
    msg = b
      ? `เลือกบอลลูน ${b.uid} (${b.kind}) — ลากย้าย · ลากมุมขวาล่างปรับขนาด · พิมพ์คำแปลด้านขวา`
      : 'คลิกบอลลูนในภาพเพื่อแก้คำแปล/ย้าย/ปรับขนาด · หรือเลือกเครื่องมือด้านบนเพื่อแปรงลบ/เพิ่มบอลลูน';
  }
  ico.textContent = icon;
  txt.textContent = msg;
}

$('toolSelect').onclick = () => setTool('select');
$('toolBrush').onclick = () => setTool('brush');
$('toolAdd').onclick = () => setTool('add');
$('brushSize').addEventListener('input', (e) => {
  state.brushSize = Number(e.target.value);
  $('brushSizeVal').textContent = e.target.value;
  moveBrushCursor(state.hover);
  updateStatus();
});
$('brushModeErase').onclick = () => toggleBrushErase();
$('btnUndoStroke').onclick = () => undoStroke();
$('btnClearBrush').onclick = () => { clearBrush(); toast('ล้างที่แปรงทั้งหมดแล้ว'); };

/* ------------------------- help modal --------------------------- */
$('btnHelp').onclick = () => $('helpModal').classList.remove('hidden');
$('btnHelpClose').onclick = () => $('helpModal').classList.add('hidden');
$('helpModal').addEventListener('click', (e) => {
  if (e.target.id === 'helpModal') e.target.classList.add('hidden');
});

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
  const opts = compressOptions();
  toast(opts.compress ? `กำลังเรนเดอร์ + บีบอัด (≤ ${opts.target_mb} MB)...` : 'กำลังเรนเดอร์ใหม่...');
  try {
    const r = await api(`/api/page/${state.page.index}/render`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(opts),
    });
    state.page.media = r.media;
    if (!state.page.media[state.view]) { state.view = 'out'; $('viewSelect').value = 'out'; }
    refreshImage(true);
    state.page.output = Object.assign({}, state.page.output, {
      exists: true,
      size_human: r.info?.size_human,
      under_target: r.info?.ok,
    });
    showOutputInfo();
    if (state.pageIndex != null) loadMetrics(state.pageIndex);
    const info = r.info || {};
    if (info.compressed) {
      toast(info.ok
        ? `เรนเดอร์ + บีบอัดแล้ว ✓ ขนาด ${info.size_human}`
        : `เรนเดอร์แล้ว แต่ยังเกินเป้า (ได้ ${info.size_human}) — ลองลดขนาดหรือย่อภาพ`,
        !info.ok);
    } else {
      toast('เรนเดอร์ใหม่แล้ว ✓');
    }
  } catch (e) { toast('เรนเดอร์ไม่สำเร็จ: ' + (e.body?.error || e.message), true); }
}

/* อ่านตัวเลือก "ขนาดไฟล์/ฟอร์แมต" จาก topbar */
function compressOptions() {
  const mb = parseFloat($('targetMb') ? $('targetMb').value : '1');
  const fmt = $('outFormat') ? $('outFormat').value : 'webp';
  const noCompress = !(mb > 0);
  return {
    compress: !noCompress,
    target_mb: noCompress ? 1 : mb,
    format: fmt,
  };
}

/* แสดงขนาดไฟล์ผลลัพธ์ปัจจุบันบนแถบสถานะ */
function showOutputInfo() {
  const el = $('outputInfo');
  if (!el) return;
  const o = state.page && state.page.output;
  el.classList.remove('good', 'bad');
  if (!o || !o.exists) { el.innerHTML = 'ยังไม่มีไฟล์ผลลัพธ์'; return; }
  const mark = o.under_target === false ? '🔴' : '🟢';
  el.innerHTML = `${mark} ไฟล์ผลลัพธ์: <b>${o.size_human || '-'}</b> / เป้า ${o.target_mb}MB`;
  el.classList.add(o.under_target === false ? 'bad' : 'good');
}

async function addBubble(bbox, erase = false) {
  if (!state.page) { toast('ยังไม่ได้เลือกหน้า', true); return; }
  try {
    toast(erase ? 'กำลังเพิ่มบอลลูน + ลบข้อความเดิม (AI)...' : 'กำลังเพิ่มบอลลูน...');
    const r = await api(`/api/page/${state.page.index}/bubbles`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ op: 'add', bbox, kind: $('newKind').value, erase: !!erase }),
    });
    await selectPage(state.page.index);
    selectBubble(r.uid);
    setTool('select');
    const ta = $('textInput'); if (ta) ta.focus();
    const bad = erase && r.erased === false;
    toast(bad ? 'เพิ่มบอลลูนแล้ว (แต่ลบข้อความเดิมไม่สำเร็จ — ลองใช้แปรงลบ)'
              : 'เพิ่มบอลลูนแล้ว ✓ พิมพ์คำแปลได้เลย', bad);
  } catch (e) { toast('เพิ่มบอลลูนไม่สำเร็จ: ' + (e.body?.error || e.message), true); }
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
    const r = await api(`/api/page/${state.page.index}/inpaint`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ op: 'inpaint', strokes: state.strokes, grow: 4 }),
    });
    clearBrush();
    refreshImage(true);
    if (state.pageIndex != null) loadMetrics(state.pageIndex);
    toast(`ลบพื้นที่เรียบร้อย ✓${r && r.method ? ' (' + r.method + ')' : ''}`);
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
    if (state.pageIndex != null) loadMetrics(state.pageIndex);
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

/* ------------------------- คีย์ลัด ------------------------------ */
document.addEventListener('keydown', (e) => {
  const tag = (document.activeElement && document.activeElement.tagName) || '';
  const typing = ['TEXTAREA', 'INPUT', 'SELECT'].includes(tag);

  if (e.key === 'Escape') {
    if (addRect) { addRect = null; ctx.clearRect(0, 0, canvas.width, canvas.height); updateStatus(); }
    const hm = $('helpModal'); if (hm && !hm.classList.contains('hidden')) hm.classList.add('hidden');
    return;
  }
  if (typing) return;
  if (e.ctrlKey || e.metaKey) {
    if (e.key.toLowerCase() === 's') { e.preventDefault(); savePage(false); }
    else if (e.key.toLowerCase() === 'z' && state.tool === 'brush') { e.preventDefault(); undoStroke(); }
    return;
  }
  if (e.key === '1') setTool('select');
  else if (e.key === '2') setTool('brush');
  else if (e.key === '3') setTool('add');
  else if (e.key.toLowerCase() === 'e' && state.tool === 'brush') toggleBrushErase();
});

/* ------------------------- ตัวเลือกขนาดไฟล์ (compress) ---------- */
async function loadOutputSettings() {
  try {
    const s = await api('/api/output-settings');
    if ($('targetMb')) $('targetMb').value = s.compress ? String(s.target_mb) : '0';
    if ($('outFormat')) $('outFormat').value = s.format || 'webp';
  } catch (e) { /* ใช้ค่าดีฟอลต์ใน HTML */ }
}
if ($('targetMb')) $('targetMb').addEventListener('change', () => {
  const mb = parseFloat($('targetMb').value);
  toast(mb > 0 ? `จะบีบอัดผลลัพธ์ให้ไม่เกิน ${mb} MB เมื่อเรนเดอร์` : 'ปิดการบีบอัด (ไฟล์เต็มขนาด)');
});

/* ------------------------- boot -------------------------------- */
setTool('select');
updateStatus();
loadOutputSettings();
loadProject().catch((e) => toast('โหลดโปรเจกต์ไม่สำเร็จ: ' + e.message, true));

/* ======================= Monitoring Dashboard =======================
 * ดึง metrics ของหน้าที่เลือกจาก /api/metrics/<index> แล้วเรนเดอร์เป็น
 * การ์ดต่อสเตจ + รายการ issues + ภาพวินิจฉัย (mask/overlay)
 * ================================================================== */

const STAGE_META = {
  detection:   { ico: '🔍', name: 'Detection (OCR+SAM)' },
  translation: { ico: '🌐', name: 'Translation (LLM)' },
  inpainting:  { ico: '🧽', name: 'Inpainting (ลบข้อความ)' },
  typesetting: { ico: '✍️', name: 'Typesetting (วาดคำไทย)' },
};

const ISSUE_SEV = {
  // error = แดง, warn = เหลือง
  NO_TEXT_DETECTED: 'error',
  OVERSIZED_BUBBLE: 'error',
  HIGH_MASK_COVERAGE: 'error',
  EMPTY_MASK: 'error',
  MISSING_TRANSLATION: 'error',
  LEFTOVER_ENGLISH: 'error',
  LOW_CONFIDENCE: 'warn',
  UNRENDERED_BUBBLES: 'warn',
};

function fmtStageKV(stage, data) {
  // แปลงข้อมูลของแต่ละสเตจเป็นแถว key->value ที่อ่านง่าย
  const rows = [];
  if (stage === 'detection') {
    rows.push(['บล็อกข้อความ', data.text_blocks]);
    rows.push(['บอลลูน', data.bubbles]);
    if (data.low_confidence_blocks) rows.push(['OCR มั่นใจต่ำ', data.low_confidence_blocks]);
    if (data.bubble_kinds) {
      const kinds = Object.entries(data.bubble_kinds).map(([k, v]) => `${k}:${v}`).join(' ');
      rows.push(['ประเภท', kinds]);
    }
  } else if (stage === 'translation') {
    rows.push(['แปลแล้ว', `${data.translated}/${data.bubbles_total}`]);
    if (data.missing) rows.push(['ขาดคำแปล', data.missing]);
    if (data.leftover_english) rows.push(['ไทยปนอังกฤษ', data.leftover_english]);
  } else if (stage === 'inpainting') {
    if (data.status === 'skipped') rows.push(['สถานะ', 'ข้าม (ไม่มี mask)']);
    else {
      rows.push(['พิกเซลที่ลบ', (data.mask_pixels ?? 0).toLocaleString()]);
      rows.push(['ครอบคลุมภาพ', `${data.coverage_pct ?? 0}%`]);
    }
  } else if (stage === 'typesetting') {
    rows.push(['วาดคำแปลแล้ว', `${data.rendered}/${data.bubbles_total}`]);
    if (data.unrendered) rows.push(['ยังไม่ได้วาด', data.unrendered]);
  }
  return rows;
}

async function loadMetrics(index) {
  const body = $('dashBody');
  const imgs = $('dashImages');
  let m = null;
  try {
    m = await api(`/api/metrics/${index}`);
  } catch (e) {
    body.innerHTML = `<p class="muted small">ยังไม่มีข้อมูล metrics ของหน้านี้<br>(จะถูกสร้างอัตโนมัติหลังประมวลผลด้วยเวอร์ชันใหม่)</p>`;
    imgs.classList.add('hidden');
    return;
  }

  let html = '';
  // การ์ดต่อสเตจ
  for (const [key, meta] of Object.entries(STAGE_META)) {
    const data = m.stages?.[key];
    if (!data) continue;
    const rows = fmtStageKV(key, data)
      .map(([k, v]) => `<div class="kv"><span>${k}</span><b>${v}</b></div>`).join('');
    html += `<div class="stage-card"><div class="stage-title"><span class="ico">${meta.ico}</span>${meta.name}</div>${rows}</div>`;
  }

  // issues
  const issues = m.issues || [];
  if (issues.length === 0) {
    html += `<div class="ok-badge">✅ ไม่พบปัญหาในหน้านี้</div>`;
  } else {
    html += `<div class="stage-card"><div class="stage-title"><span class="ico">⚠️</span>พบ ${issues.length} ประเด็น</div>` +
      issues.map((it) => {
        const sev = ISSUE_SEV[it.code] === 'warn' ? 'warn' : '';
        return `<div class="issue ${sev}"><div class="code">${it.code}</div>${it.message}</div>`;
      }).join('') + '</div>';
  }
  body.innerHTML = html;

  // ภาพวินิจฉัย
  const hasMask = m.artifacts?.mask;
  const hasOverlay = m.artifacts?.overlay;
  if (hasMask || hasOverlay) {
    imgs.classList.remove('hidden');
    if (hasMask) {
      $('imgMask').src = `/monitoring/${index}/${hasMask}`;
      $('linkMask').href = `/monitoring/${index}/${hasMask}`;
    }
    if (hasOverlay) {
      $('imgOverlay').src = `/monitoring/${index}/${hasOverlay}`;
      $('linkOverlay').href = `/monitoring/${index}/${hasOverlay}`;
    }
  } else {
    imgs.classList.add('hidden');
  }
}

// ปุ่มรีเฟรช metrics
document.addEventListener('DOMContentLoaded', () => {
  const btn = $('btnRefreshMetrics');
  if (btn) btn.onclick = () => state.pageIndex != null && loadMetrics(state.pageIndex);
});
