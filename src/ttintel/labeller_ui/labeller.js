(() => {
  const canvas = document.getElementById('frame-canvas');
  const ctx = canvas.getContext('2d');
  const state = { frameIds: [], index: 0, image: null, meta: null, zoom: 1, focus: { x: 0, y: 0 }, mode: 'ball', calibration: false, manual: true, tracker: false, cornerClicks: [] };
  const $ = (id) => document.getElementById(id);
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

  function currentId() { return state.frameIds[state.index]; }
  function currentRecord() { return state.meta; }
  function setAction(message, tone = 'human') { $('last-action').textContent = message; $('last-action').style.color = `var(--${tone})`; }
  function pointList(points) { return (points || []).map((p) => ({ x: Number(p.x), y: Number(p.y) })); }

  function viewBox() {
    if (!state.image) return { x: 0, y: 0, w: 1, h: 1 };
    const width = state.image.naturalWidth;
    const height = state.image.naturalHeight;
    const w = width / state.zoom;
    const h = height / state.zoom;
    return { x: clamp(state.focus.x - w / 2, 0, width - w), y: clamp(state.focus.y - h / 2, 0, height - h), w, h };
  }

  function drawPolyline(points, colour, dashed = false) {
    if (!points || points.length < 2) return;
    const box = viewBox();
    ctx.save(); ctx.strokeStyle = colour; ctx.lineWidth = 3; ctx.setLineDash(dashed ? [9, 7] : []); ctx.beginPath();
    points.forEach((point, i) => { const x = (point.x - box.x) / box.w * canvas.width; const y = (point.y - box.y) / box.h * canvas.height; if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    if (points.length > 2) ctx.closePath(); ctx.stroke(); ctx.restore();
  }

  function drawPoint(point, colour, label, radius = 10) {
    if (!point) return;
    const box = viewBox(); const x = (point.x - box.x) / box.w * canvas.width; const y = (point.y - box.y) / box.h * canvas.height;
    ctx.save(); ctx.strokeStyle = colour; ctx.fillStyle = 'rgba(16,14,32,.7)'; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = colour; ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
    if (label) { ctx.font = '700 12px system-ui'; ctx.fillStyle = colour; ctx.fillText(label, x + radius + 5, y - radius - 2); }
    ctx.restore();
  }

  function render() {
    if (!state.image) return;
    const box = viewBox(); ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.drawImage(state.image, box.x, box.y, box.w, box.h, 0, 0, canvas.width, canvas.height);
    const meta = currentRecord();
    if (state.calibration) drawPolyline(meta.auto_corners, '#a78bfa', true);
    if (state.manual) {
      drawPolyline(meta.manual_corners, '#ffc56b');
      if (state.cornerClicks.length) drawPolyline(state.cornerClicks, '#ffc56b');
      if (meta.label?.label === 'point') drawPoint({ x: meta.label.x, y: meta.label.y }, '#ffc56b', 'HUMAN');
      if (meta.label?.label === 'absent') { ctx.save(); ctx.fillStyle = 'rgba(255,197,107,.9)'; ctx.font = '800 13px system-ui'; ctx.fillText('NO BALL', 18, 28); ctx.restore(); }
    }
    if (state.tracker && meta.tracker?.point) drawPoint(meta.tracker.point, '#47d9c0', `TRACKER ${Math.round(meta.tracker.confidence * 100)}%`, 8);
    $('frame-id').textContent = `Frame ${currentId()}`; $('timestamp').textContent = `${Number(meta.timestamp).toFixed(3)} s`;
    $('frame-count').textContent = `${state.index + 1} / ${state.frameIds.length}`; $('scrubber').value = state.index;
  }

  async function fetchMeta() {
    const response = await fetch(`/api/frame/${currentId()}?calibration=${state.calibration ? 1 : 0}`);
    if (!response.ok) throw new Error('frame metadata unavailable');
    state.meta = await response.json(); render(); updateHint();
  }

  async function loadFrame(index) {
    state.index = clamp(Number(index), 0, state.frameIds.length - 1); const id = currentId();
    const image = new Image(); image.src = `/api/frame/${id}/image`; await image.decode(); state.image = image;
    state.focus = { x: image.naturalWidth / 2, y: image.naturalHeight / 2 }; await fetchMeta();
  }

  function updateHint() { $('canvas-hint').textContent = state.mode === 'corners' ? `Corner ${state.cornerClicks.length + 1} of 4 · near-left → near-right → far-right → far-left` : 'Click the ball · press N when it is not visible'; $('mode-pill').textContent = state.mode.toUpperCase(); }
  function updateCoverage(counts) { const total = state.frameIds.length; const percent = total ? Math.round(counts.labelled / total * 100) : 0; $('coverage-percent').textContent = `${percent}%`; $('coverage-bar').style.width = `${percent}%`; $('labelled-count').textContent = counts.labelled; $('point-count').textContent = counts.point; $('absent-count').textContent = counts.absent; $('untouched-count').textContent = counts.untouched; }

  async function saveLabel(label, point) {
    const payload = { frame_id: currentId(), label }; if (point) Object.assign(payload, point);
    const response = await fetch('/api/labels', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }); const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'could not save label'); state.meta = result.frame; const session = await (await fetch('/api/session')).json(); updateCoverage(session.counts); render();
  }

  async function saveCorners() {
    const response = await fetch('/api/corners', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ corners: state.cornerClicks }) }); const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'could not save corners'); setAction('Four corners saved · loadable by --manual-corners'); state.cornerClicks = []; await fetchMeta(); updateHint();
  }

  function imagePoint(event) { const rect = canvas.getBoundingClientRect(); const px = (event.clientX - rect.left) / rect.width * canvas.width; const py = (event.clientY - rect.top) / rect.height * canvas.height; const box = viewBox(); return { x: clamp(box.x + px / canvas.width * box.w, 0, state.image.naturalWidth), y: clamp(box.y + py / canvas.height * box.h, 0, state.image.naturalHeight) }; }

  canvas.addEventListener('mousemove', (event) => { if (!state.image) return; const point = imagePoint(event); state.focus = point; render(); });
  canvas.addEventListener('click', async (event) => {
    try {
      const point = imagePoint(event);
      if (state.mode === 'corners') { state.cornerClicks.push(point); render(); if (state.cornerClicks.length === 4) { await saveCorners(); setAction('Four corners saved · loadable by --manual-corners'); } else { updateHint(); setAction(`Corner ${state.cornerClicks.length} recorded`); } }
      else { await saveLabel('point', point); setAction(`Ball saved at ${point.x.toFixed(1)}, ${point.y.toFixed(1)}`); }
    } catch (error) { setAction(error.message, 'danger'); }
  });
  $('absent-button').addEventListener('click', async () => { try { await saveLabel('absent'); setAction('Explicit “no ball” label saved'); } catch (error) { setAction(error.message, 'danger'); } });
  $('scrubber').addEventListener('input', (event) => loadFrame(event.target.value).catch((error) => setAction(error.message, 'danger')));
  $('prev-frame').addEventListener('click', () => loadFrame(state.index - 1)); $('next-frame').addEventListener('click', () => loadFrame(state.index + 1));
  $('zoom').addEventListener('input', (event) => { state.zoom = Number(event.target.value); $('zoom-value').textContent = `${state.zoom}×`; render(); });
  document.querySelectorAll('.mode').forEach((button) => button.addEventListener('click', () => { document.querySelectorAll('.mode').forEach((item) => item.classList.remove('active')); button.classList.add('active'); state.mode = button.dataset.mode; state.cornerClicks = []; $('absent-button').style.display = state.mode === 'corners' ? 'none' : 'block'; $('instruction').textContent = state.mode === 'corners' ? 'Click the table corners in the numbered order shown on the canvas.' : 'Place the marker at the ball centre. Click again to replace.'; updateHint(); }));
  $('calibration-toggle').addEventListener('change', async (event) => { state.calibration = event.target.checked; await fetchMeta(); });
  $('manual-toggle').addEventListener('change', (event) => { state.manual = event.target.checked; render(); });
  $('tracker-button').addEventListener('click', async () => { const button = $('tracker-button'); button.disabled = true; button.textContent = 'Running…'; $('tracker-status').textContent = 'Sequential pass in progress; this can take a moment.'; try { const response = await fetch('/api/tracker-overlay', { method: 'POST' }); const result = await response.json(); if (!response.ok) throw new Error(result.message || 'tracker unavailable'); state.tracker = true; button.textContent = 'Ready'; $('tracker-status').textContent = `${result.count} frame outputs cached from one sequential pass.`; await fetchMeta(); } catch (error) { button.disabled = false; button.textContent = 'Retry'; $('tracker-status').textContent = error.message; } });
  document.addEventListener('keydown', (event) => { if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return; if (event.key === 'ArrowLeft') { event.preventDefault(); loadFrame(state.index - 1); } else if (event.key === 'ArrowRight') { event.preventDefault(); loadFrame(state.index + 1); } else if (event.key.toLowerCase() === 'n' && state.mode === 'ball') { event.preventDefault(); $('absent-button').click(); } });

  (async () => { try { const response = await fetch('/api/session'); const session = await response.json(); state.frameIds = session.frame_ids; $('video-name').textContent = session.video; $('scrubber').max = Math.max(0, state.frameIds.length - 1); $('labels-path').textContent = session.labels_path; $('corners-path').textContent = session.corners_path; updateCoverage(session.counts); if (session.tracker.status === 'ready') { state.tracker = true; $('tracker-button').textContent = 'Ready'; } await loadFrame(0); } catch (error) { setAction(error.message, 'danger'); } })();
})();
