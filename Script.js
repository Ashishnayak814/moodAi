/**
 * MoodAI – Real-time Emotion Detection (FIXED v3)
 * Root cause fixed: canvas pixel size vs CSS display size mismatch
 * bbox scale = (canvas CSS display size) / (captured frame size)
 */

const API_URL        = "http://127.0.0.1:8000/predict";
const BACKEND_HEALTH = "http://127.0.0.1:8000/health";
const JPEG_Q         = 0.70;
const SEND_COOLDOWN_MS = 80;

// ─── DOM refs ────────────────────────────────────────────────────────────────
const videoEl       = document.getElementById("webcam");
const canvasEl      = document.getElementById("overlay");
const ctx           = canvasEl.getContext("2d");
const startBtn      = document.getElementById("btn-start");
const stopBtn       = document.getElementById("btn-stop");
const statusDot     = document.getElementById("status-dot");
const statusText    = document.getElementById("status-text");
const emotionLabel  = document.getElementById("emotion-label");
const emotionEmoji  = document.getElementById("emotion-emoji");
const confidencePct = document.getElementById("confidence-pct");
const confidenceBar = document.getElementById("confidence-bar");
const noFaceMsg     = document.getElementById("no-face-msg");
const faceCountEl   = document.getElementById("face-count");
const fpsCounterEl  = document.getElementById("fps-counter");

const breakdownBars = {
  Angry:     document.getElementById("bar-angry"),
  Disgust:   document.getElementById("bar-disgust"),
  Fear:      document.getElementById("bar-fear"),
  Happy:     document.getElementById("bar-happy"),
  Neutral:   document.getElementById("bar-neutral"),
  Sad:       document.getElementById("bar-sad"),
  Surprised: document.getElementById("bar-surprised"),
};
const breakdownPcts = {
  Angry:     document.getElementById("pct-angry"),
  Disgust:   document.getElementById("pct-disgust"),
  Fear:      document.getElementById("pct-fear"),
  Happy:     document.getElementById("pct-happy"),
  Neutral:   document.getElementById("pct-neutral"),
  Sad:       document.getElementById("pct-sad"),
  Surprised: document.getElementById("pct-surprised"),
};

const EMOJIS = {
  Angry:"😠", Disgust:"🤢", Fear:"😨",
  Happy:"😊", Neutral:"😐", Sad:"😢", Surprised:"😲",
};
const BOX_COLORS = {
  Angry:"#ff4444", Disgust:"#a855f7", Fear:"#f97316",
  Happy:"#22c55e", Neutral:"#94a3b8", Sad:"#60a5fa", Surprised:"#facc15",
};

// ─── State ───────────────────────────────────────────────────────────────────
let stream            = null;
let predicting        = false;
let loopTimer         = null;
let captureCanvas, captureCtx;
let captureW = 640, captureH = 480;

let fpsCount = 0, fpsLast = performance.now();
let backendHealthy = false, modelLoaded = false, predictionEnabled = false;
let retryCount = 0;

// ─── THE KEY FIX: get actual CSS rendered size of canvas ─────────────────────
// canvasEl.width  = pixel buffer size (e.g. 640)
// canvasEl.getBoundingClientRect() = actual screen size (e.g. 820×615)
// bbox from backend is in captureW×captureH space
// We must scale to CSS display size
function getScales() {
  const rect = canvasEl.getBoundingClientRect();
  return {
    x: rect.width  / captureW,
    y: rect.height / captureH,
  };
}

// ─── Status ──────────────────────────────────────────────────────────────────
function setStatus(state) {
  const cfg = {
    idle:    { color:"#64748b", text:"IDLE",           animate:false },
    loading: { color:"#facc15", text:"INITIALISING …", animate:true  },
    active:  { color:"#22c55e", text:"LIVE",           animate:true  },
    error:   { color:"#f87171", text:"ERROR",          animate:false },
  }[state] || { color:"#64748b", text:"IDLE", animate:false };

  statusDot.style.backgroundColor = cfg.color;
  statusDot.style.boxShadow = `0 0 12px ${cfg.color}80`;
  statusDot.style.animation = cfg.animate ? "pulse 1.2s infinite" : "none";
  statusText.textContent    = cfg.text;

  const badge = document.getElementById("model-badge");
  if (badge) {
    badge.textContent   = "● AI MODEL ACTIVE";
    badge.style.opacity = state === "active" ? "1" : "0.35";
  }
}

// ─── Draw bounding boxes ──────────────────────────────────────────────────────
function drawBoundingBoxes(faces) {
  // Canvas pixel buffer = CSS display size (set in syncCanvasSize)
  // So 1 canvas pixel = 1 CSS pixel — no extra scaling needed here
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  if (!faces || faces.length === 0) return;

  // Scale factors: backend coords (captureW×captureH) → canvas pixels (CSS size)
  const scaleX = canvasEl.width  / captureW;
  const scaleY = canvasEl.height / captureH;

  faces.forEach(face => {
    const [fx, fy, fw, fh] = face.bbox;
    const x = fx * scaleX;
    const y = fy * scaleY;
    const w = fw * scaleX;
    const h = fh * scaleY;
    const color = BOX_COLORS[face.emotion] || "#00f5ff";

    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur  = 18;
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2.5;
    ctx.strokeRect(x, y, w, h);

    // Corner accents
    const cs = 16;
    ctx.lineWidth  = 3.5;
    ctx.shadowBlur = 0;
    [
      [x,     y,      cs,  cs ],
      [x + w, y,     -cs,  cs ],
      [x,     y + h,  cs, -cs ],
      [x + w, y + h, -cs, -cs ],
    ].forEach(([px, py, dx, dy]) => {
      ctx.beginPath();
      ctx.moveTo(px + dx, py);
      ctx.lineTo(px, py);
      ctx.lineTo(px, py + dy);
      ctx.stroke();
    });

    // Label pill — flip below box if near top edge
    const label = `${face.emotion.toUpperCase()}  ${face.confidence.toFixed(1)}%`;
    ctx.font = "bold 13px 'Inter', sans-serif";
    const tw = ctx.measureText(label).width;
    const lx = x;
    const ly = y > 30 ? y - 28 : y + h + 6;

    ctx.fillStyle = color + "cc";
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(lx, ly, tw + 18, 24, 6);
    else               ctx.rect(lx, ly, tw + 18, 24);
    ctx.fill();

    ctx.shadowBlur = 0;
    ctx.fillStyle  = "#000";
    ctx.fillText(label, lx + 9, ly + 17);
    ctx.restore();
  });
}

// ─── Sync canvas pixel buffer to its CSS display size ────────────────────────
// This is called once on start and on window resize.
// Without this, canvas.width=640 but element shows at 820px → everything squished.
function syncCanvasSize() {
  const rect = canvasEl.getBoundingClientRect();
  if (rect.width > 0 && rect.height > 0) {
    canvasEl.width  = Math.round(rect.width);
    canvasEl.height = Math.round(rect.height);
  }
}

// ─── Analytics panel ─────────────────────────────────────────────────────────
function updateAnalyticsPanel(faces) {
  const videoEmotionLabel = document.getElementById("video-emotion-label");

  if (!faces || faces.length === 0) {
    noFaceMsg.classList.remove("hidden");
    emotionLabel.textContent  = "—";
    emotionEmoji.textContent  = "🔍";
    confidencePct.textContent = "0%";
    confidenceBar.style.width = "0%";
    faceCountEl.textContent   = "0";
    if (videoEmotionLabel) videoEmotionLabel.textContent = "No face";
    resetBreakdownBars();
    return;
  }

  noFaceMsg.classList.add("hidden");
  faceCountEl.textContent = faces.length;

  const top = faces.reduce((a, b) => a.confidence > b.confidence ? a : b);
  emotionLabel.textContent  = top.emotion.toUpperCase();
  emotionEmoji.textContent  = EMOJIS[top.emotion] || "🙂";
  confidencePct.textContent = `${top.confidence.toFixed(1)}%`;
  confidenceBar.style.width = `${top.confidence}%`;
  if (videoEmotionLabel)
    videoEmotionLabel.textContent = `${top.emotion.toUpperCase()} (${top.confidence.toFixed(0)}%)`;

  const scores = top.all_scores || {};
  Object.keys(breakdownBars).forEach(e => {
    const pct = scores[e] ?? 0;
    if (breakdownBars[e]) breakdownBars[e].style.height = `${pct}%`;
    if (breakdownPcts[e]) breakdownPcts[e].textContent  = `${pct.toFixed(1)}%`;
  });
}

function resetBreakdownBars() {
  Object.keys(breakdownBars).forEach(e => {
    if (breakdownBars[e]) breakdownBars[e].style.height = "0%";
    if (breakdownPcts[e]) breakdownPcts[e].textContent  = "0%";
  });
}

// ─── FPS ─────────────────────────────────────────────────────────────────────
function tickFPS() {
  fpsCount++;
  const now = performance.now();
  if (now - fpsLast >= 1000) {
    fpsCounterEl.textContent = Math.round((fpsCount / (now - fpsLast)) * 1000);
    fpsCount = 0;
    fpsLast  = now;
  }
}

// ─── Backend health ───────────────────────────────────────────────────────────
async function checkBackendHealth() {
  try {
    const r    = await fetch(BACKEND_HEALTH, { cache: "no-store" });
    const data = await r.json();
    backendHealthy    = data.status === "ok";
    modelLoaded       = data.model_loaded === true;
    predictionEnabled = backendHealthy && modelLoaded;
  } catch {
    backendHealthy = modelLoaded = predictionEnabled = false;
  }
}

// ─── Prediction loop ─────────────────────────────────────────────────────────
function scheduleNext() {
  if (!stream || !predictionEnabled) return;
  loopTimer = setTimeout(sendFrame, SEND_COOLDOWN_MS);
}

function sendFrame() {
  if (!stream || !predictionEnabled || videoEl.readyState < 2) {
    scheduleNext();
    return;
  }
  if (predicting) {
    loopTimer = setTimeout(sendFrame, 20);
    return;
  }

  predicting = true;
  captureCtx.drawImage(videoEl, 0, 0, captureW, captureH);
  const dataURL = captureCanvas.toDataURL("image/jpeg", JPEG_Q);

  fetch(API_URL, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ image: dataURL }),
  })
    .then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      retryCount = 0;
      return r.json();
    })
    .then(data => {
      if (data.success) {
        drawBoundingBoxes(data.faces);
        updateAnalyticsPanel(data.faces);
        tickFPS();
      }
    })
    .catch(err => {
      console.warn("[MoodAI]", err);
      if (++retryCount >= 5) setStatus("error");
    })
    .finally(() => {
      predicting = false;
      scheduleNext();
    });
}

// ─── Camera start/stop ────────────────────────────────────────────────────────
async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    alert("Camera API unavailable. Open via http://localhost:8000/ in Chrome/Firefox.");
    return;
  }
  setStatus("loading");
  startBtn.disabled = true;

  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    videoEl.srcObject = stream;
    await videoEl.play();

    // Wait for video metadata so videoWidth/Height are populated
    await new Promise(r => {
      if (videoEl.readyState >= 1) return r();
      videoEl.addEventListener("loadedmetadata", r, { once: true });
    });

    captureW = videoEl.videoWidth  || 640;
    captureH = videoEl.videoHeight || 480;

    // KEY FIX: set canvas pixel buffer = its CSS display size (not video res)
    // This must happen AFTER the video is visible so getBoundingClientRect works
    document.getElementById("cam-placeholder")?.classList.add("hidden");
    videoEl.style.cssText += ";display:block;visibility:visible;opacity:1;";

    // Small delay so layout reflows and CSS size is correct
    await new Promise(r => setTimeout(r, 50));
    syncCanvasSize();

    // Offscreen canvas matches video resolution (what backend gets)
    captureCanvas        = document.createElement("canvas");
    captureCanvas.width  = captureW;
    captureCanvas.height = captureH;
    captureCtx           = captureCanvas.getContext("2d");

    stopBtn.disabled  = false;
    startBtn.disabled = true;
    setStatus("active");

    // Re-sync on window resize so boxes stay accurate
    window.addEventListener("resize", syncCanvasSize);

    await checkBackendHealth();
    predicting = false;
    retryCount = 0;
    sendFrame();

  } catch (err) {
    setStatus("error");
    startBtn.disabled = false;
    let msg = `Camera error: ${err.message}`;
    if (err.name === "NotAllowedError") msg += "\n\nAllow camera permission in browser settings.";
    if (err.name === "NotFoundError")   msg += "\n\nNo webcam found.";
    alert(msg);
  }
}

function stopCamera() {
  clearTimeout(loopTimer);
  loopTimer  = null;
  predicting = false;

  window.removeEventListener("resize", syncCanvasSize);

  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  videoEl.srcObject = null;
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  setStatus("idle");
  startBtn.disabled = false;
  stopBtn.disabled  = true;
  document.getElementById("cam-placeholder")?.classList.remove("hidden");
  updateAnalyticsPanel(null);
  faceCountEl.textContent  = "0";
  fpsCounterEl.textContent = "0";
}

// ─── Init ─────────────────────────────────────────────────────────────────────
startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click",  stopCamera);
setStatus("idle");
stopBtn.disabled = true;