(() => {
  const canvas = document.getElementById('frame-canvas');
  const ctx = canvas.getContext('2d');
  const state = {
    frameIds: [], index: 0, image: null, meta: null, zoom: 1,
    focus: { x: 0, y: 0 }, mode: 'ball', overlay: 'none', cornerClicks: [],
    tracker: { status: 'off', message: '' }
  };
  const $ = (id) => document.getElementById(id);
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const MODEL = '#ff0000';
  const HUMAN = '#0000ff';

  function currentId() { return state.frameIds[state.index]; }
  function setAction(message) { $('last-action').textContent = message; }
  function viewBox() {
    if (!state.image) return { x: 0, y: 0, w: 1, h: 1 };
    const width = state.image.naturalWidth;
    const height = state.image.naturalHeight;
    const w = width / state.zoom;
    const h = height / state.zoom;
    return {
      x: clamp(state.focus.x - w / 2, 0, width - w),
      y: clamp(state.focus.y - h / 2, 0, height - h), w, h
    };
  }
  function canvasPoint(point) {
    const box = viewBox();
    return { x: (point.x - box.x) / box.w * canvas.width, y: (point.y - box.y) / box.h * canvas.height };
  }
  function drawPolyline(points, colour, width, dashed, label) {
    if (!points || points.length < 2) return;
    ctx.save();
    ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.setLineDash(dashed ? [8, 6] : []);
    ctx.beginPath();
    points.forEach((point, index) => { const p = canvasPoint(point); if (index === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y); });
    if (points.length > 2) ctx.closePath();
    ctx.stroke(); ctx.restore();
    if (label) { const p = canvasPoint(points[0]); drawChip(label, p.x + 5, p.y + 5, colour); }
  }
  function drawChip(text, x, y, colour) {
    ctx.save();
    ctx.font = '12px Arial, Helvetica, sans-serif';
    const width = ctx.measureText(text).width + 8;
    const top = clamp(y, 0, canvas.height - 20);
    const left = clamp(x, 0, canvas.width - width);
    ctx.fillStyle = colour; ctx.fillRect(left, top, width, 18);
    ctx.fillStyle = '#fff'; ctx.fillText(text, left + 4, top + 13);
    ctx.restore();
  }
  function drawBall(point, colour, label, radius) {
    if (!point) return;
    const p = canvasPoint(point);
    ctx.save(); ctx.strokeStyle = colour; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(p.x, p.y, radius, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
    drawChip(label, p.x + radius + 5, p.y - radius - 4, colour);
  }
  function render() {
    if (!state.image || !state.meta) return;
    const box = viewBox();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(state.image, box.x, box.y, box.w, box.h, 0, 0, canvas.width, canvas.height);
    const meta = state.meta;
    const showTable = state.overlay === 'detections' || state.overlay === 'table';
    const showBall = state.overlay === 'detections' || state.overlay === 'ball';
    if (showTable && meta.auto_corners) {
      const sensitivity = meta.auto_corner_sensitivity_m_per_px;
      const label = sensitivity == null ? 'corner_sensitivity_m_per_px unavailable' : `corner_sensitivity_m_per_px ${sensitivity.toFixed(4)}`;
      drawPolyline(meta.auto_corners, MODEL, 2, true, label);
    }
    if (showBall && meta.tracker && meta.tracker.point) {
      const confidence = Number(meta.tracker.confidence || 0).toFixed(3);
      drawBall(meta.tracker.point, MODEL, `${meta.tracker.source || 'ball'} ${confidence}`, 9);
    }
    if (meta.manual_corners) drawPolyline(meta.manual_corners, HUMAN, 2, false, 'human corners');
    if (state.cornerClicks.length) drawPolyline(state.cornerClicks, HUMAN, 2, false, `human corner ${state.cornerClicks.length}/4`);
    if (meta.label?.label === 'point') drawBall({ x: meta.label.x, y: meta.label.y }, HUMAN, 'human ball', 10);
    if (meta.label?.label === 'absent') drawChip('human: no ball', 10, 10, HUMAN);
    $('frame-id').textContent = currentId();
    $('frame-count').textContent = `${state.index + 1} / ${state.frameIds.length}`;
    $('scrubber').value = state.index;
  }
  async function fetchMeta() {
    const wantsTable = state.overlay === 'detections' || state.overlay === 'table';
    const response = await fetch(`/api/frame/${currentId()}?calibration=${wantsTable ? 1 : 0}`);
    if (!response.ok) throw new Error('frame metadata unavailable');
    state.meta = await response.json();
    state.tracker = { status: state.meta.tracker_status, message: state.meta.tracker_message };
    render(); updateHint(); updateOverlayStatus();
  }
  async function loadFrame(index) {
    state.index = clamp(Number(index), 0, state.frameIds.length - 1);
    const image = new Image(); image.src = `/api/frame/${currentId()}/image`; await image.decode();
    state.image = image; state.focus = { x: image.naturalWidth / 2, y: image.naturalHeight / 2 }; await fetchMeta();
  }
  function updateHint() {
    $('canvas-hint').textContent = state.mode === 'corners'
      ? `Corner ${state.cornerClicks.length + 1} of 4: near-left, near-right, far-right, far-left`
      : 'Click the ball; press N when it is not visible';
  }
  function updateOverlayStatus() {
    if (state.overlay !== 'detections' && state.overlay !== 'ball') return;
    let message;
    if (state.tracker.status === 'ready') {
      message = state.meta && state.meta.tracker && state.meta.tracker.point
        ? 'Tracker pass computed; ball detection shown in red.'
        : 'Tracker pass computed; no ball detection for this frame.';
    } else if (state.tracker.status === 'running') {
      message = 'Tracker pass is running; wait for it to finish.';
    } else if (state.tracker.status === 'unavailable') {
      message = `Tracker unavailable: ${state.tracker.message || 'click retry tracker.'}`;
    } else {
      message = 'Tracker not computed. Click “compute tracker” to enable this overlay.';
    }
    setAction(message);
  }
  function updateCoverage(counts) {
    $('labelled-count').textContent = counts.labelled; $('point-count').textContent = counts.point;
    $('absent-count').textContent = counts.absent; $('untouched-count').textContent = counts.untouched;
  }
  async function saveLabel(label, point) {
    const payload = { frame_id: currentId(), label }; if (point) Object.assign(payload, point);
    const response = await fetch('/api/labels', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const result = await response.json(); if (!response.ok) throw new Error(result.error || 'could not save label');
    state.meta = result.frame; updateCoverage((await (await fetch('/api/session')).json()).counts); render(); updateOverlayStatus();
  }
  async function clearLabel() {
    const response = await fetch('/api/labels/clear', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ frame_id: currentId() }) });
    const result = await response.json(); if (!response.ok) throw new Error(result.error || 'could not clear label');
    state.meta = result.frame; updateCoverage((await (await fetch('/api/session')).json()).counts); render(); updateOverlayStatus();
    setAction(result.cleared ? 'Current frame label cleared; frame is untouched.' : 'Current frame was already untouched.');
  }
  async function undoLabel() {
    const response = await fetch('/api/labels/undo', { method: 'POST' });
    const result = await response.json(); if (!response.ok) throw new Error(result.error || 'could not undo label');
    updateCoverage((await (await fetch('/api/session')).json()).counts);
    if (result.frame_id === currentId()) state.meta = result.frame; else await fetchMeta();
    render(); updateOverlayStatus();
    setAction(`Undid the last label on frame ${result.frame_id}.`);
  }
  async function saveCorners() {
    const response = await fetch('/api/corners', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ corners: state.cornerClicks }) });
    const result = await response.json(); if (!response.ok) throw new Error(result.error || 'could not save corners');
    state.cornerClicks = []; await fetchMeta(); setAction('Four corners saved; loadable by --manual-corners.');
  }
  function imagePoint(event) {
    const rect = canvas.getBoundingClientRect(); const px = (event.clientX - rect.left) / rect.width * canvas.width; const py = (event.clientY - rect.top) / rect.height * canvas.height; const box = viewBox();
    return { x: clamp(box.x + px / canvas.width * box.w, 0, state.image.naturalWidth), y: clamp(box.y + py / canvas.height * box.h, 0, state.image.naturalHeight) };
  }
  canvas.addEventListener('mousemove', (event) => { if (state.image && state.zoom > 1) { state.focus = imagePoint(event); render(); } });
  canvas.addEventListener('click', async (event) => {
    try {
      const point = imagePoint(event);
      if (state.mode === 'corners') {
        state.cornerClicks.push(point); render();
        if (state.cornerClicks.length === 4) await saveCorners(); else setAction(`Corner ${state.cornerClicks.length} recorded.`);
      } else { await saveLabel('point', point); setAction(`Human ball saved at ${point.x.toFixed(1)}, ${point.y.toFixed(1)}.`); }
    } catch (error) { setAction(error.message); }
  });
  $('absent-button').addEventListener('click', async () => { try { await saveLabel('absent'); setAction('Explicit human no-ball label saved.'); } catch (error) { setAction(error.message); } });
  $('clear-button').addEventListener('click', async () => { try { await clearLabel(); } catch (error) { setAction(error.message); } });
  $('undo-button').addEventListener('click', async () => { try { await undoLabel(); } catch (error) { setAction(error.message); } });
  $('scrubber').addEventListener('input', (event) => loadFrame(event.target.value).catch((error) => setAction(error.message)));
  $('prev-frame').addEventListener('click', () => loadFrame(state.index - 1)); $('next-frame').addEventListener('click', () => loadFrame(state.index + 1));
  $('zoom').addEventListener('input', (event) => { state.zoom = Number(event.target.value); $('zoom-value').textContent = `${state.zoom}x`; render(); });
  $('overlay-select').addEventListener('change', () => { state.overlay = $('overlay-select').value; updateOverlayStatus(); fetchMeta().catch((error) => setAction(error.message)); });
  $('mode-select').addEventListener('change', () => { state.mode = $('mode-select').value; state.cornerClicks = []; $('absent-button').style.display = state.mode === 'corners' ? 'none' : ''; $('instruction').textContent = state.mode === 'corners' ? 'Click table corners in the numbered order shown on the frame.' : 'Place the label at the ball centre. Click again to replace.'; updateHint(); render(); });
  $('tracker-button').addEventListener('click', async () => {
    const button = $('tracker-button'); button.disabled = true; button.textContent = 'running...'; $('tracker-status').textContent = 'sequential pass in progress'; state.tracker = { status: 'running', message: 'sequential pass in progress' }; updateOverlayStatus();
    try { const response = await fetch('/api/tracker-overlay', { method: 'POST' }); const result = await response.json(); state.tracker = { status: result.status, message: result.message || '' }; if (!response.ok) throw new Error(result.message || 'tracker unavailable'); button.textContent = 'tracker ready'; $('tracker-status').textContent = `${result.count} cached frame outputs`; await fetchMeta(); }
    catch (error) { button.disabled = false; button.textContent = 'retry tracker'; $('tracker-status').textContent = error.message; updateOverlayStatus(); }
  });
  document.addEventListener('keydown', (event) => { if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return; if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') { event.preventDefault(); $('undo-button').click(); } else if (event.key === 'ArrowLeft') { event.preventDefault(); loadFrame(state.index - 1); } else if (event.key === 'ArrowRight') { event.preventDefault(); loadFrame(state.index + 1); } else if (event.key.toLowerCase() === 'n' && state.mode === 'ball') { event.preventDefault(); $('absent-button').click(); } });
  (async () => {
    try {
      const session = await (await fetch('/api/session')).json(); state.frameIds = session.frame_ids; state.tracker = session.tracker; $('video-name').textContent = session.video; $('total-frames').textContent = state.frameIds.length; $('scrubber').max = Math.max(0, state.frameIds.length - 1); $('labels-path').textContent = session.labels_path; $('corners-path').textContent = session.corners_path; updateCoverage(session.counts); await loadFrame(0);
    } catch (error) { setAction(error.message); }
  })();
})();
