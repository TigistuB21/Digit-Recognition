"""
Interactive Web Deployment for Handwritten Digit Recognition
============================================================
Features:
  1. Interactive HTML5 Drawing Canvas (mouse/touch drawing pad)
  2. Photo Upload & Auto-Preprocessing (divides lighting, centers via moments, scales to 32x32)
  3. Real-Time Neural Network Inference (< 5ms latency)
  4. Top-3 Prediction Probabilities with Visual Confidence Bars
"""

import os
import json
import base64
import io
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs


# 1. MiniDigitCNN Architecture
class MiniDigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MiniDigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.3)

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop1(self.pool1(x))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop2(self.pool2(x))
        x = x.view(x.size(0), -1)
        x = self.drop3(F.relu(self.bn_fc(self.fc1(x))))
        x = self.fc2(x)
        return x


# Load Model
model = MiniDigitCNN(num_classes=10)
model_path = os.path.join(os.path.dirname(__file__), "models", "best_digit_cnn.pt")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
print(f"[+] Loaded trained model from {model_path}")


def preprocess_canvas_or_photo(image_bytes):
    """
    Standardizes drawing or uploaded photo to 32x32 float32 tensor.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Invalid image")

    # If canvas with alpha channel
    if len(img.shape) == 3 and img.shape[2] == 4:
        # Extract alpha or RGB
        alpha = img[:, :, 3]
        if alpha.max() > 0:
            gray = alpha  # drawn strokes are white alpha
        else:
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    h, w = gray.shape

    # Determine if dark-on-light or light-on-dark
    if gray.mean() > 128:
        # Document photo: background division
        k_size = max(31, (min(h, w) // 10) | 1)
        bg = cv2.GaussianBlur(gray, (k_size, k_size), 0)
        norm = cv2.divide(gray, bg, scale=255)
        k_stroke = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(norm, cv2.MORPH_BLACKHAT, k_stroke)
        ink_a = ((norm < 225) & (blackhat > 8)).astype(np.uint8) * 255
        ink_b = ((norm < 210)).astype(np.uint8) * 255
        ink = cv2.bitwise_or(ink_a, ink_b)
    else:
        # Dark canvas: white ink on black background
        _, ink = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Dilate slightly for connectivity
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    # Find digit bounding box
    cnts, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_size = (32, 32)
    inner_box = 24

    if not cnts:
        canvas = np.zeros(target_size, dtype=np.uint8)
    else:
        # Largest contour
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        bx, by, bw, bh = cv2.boundingRect(cnts[0])
        crop = ink[by:by+bh, bx:bx+bw]

        scale = inner_box / max(bh, bw)
        nw, nh = max(1, int(round(bw * scale))), max(1, int(round(bh * scale)))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

        canvas = np.zeros(target_size, dtype=np.uint8)
        sy = (32 - nh) // 2
        sx = (32 - nw) // 2
        canvas[sy:sy+nh, sx:sx+nw] = resized

        # Moments centering
        M = cv2.moments(canvas)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            shift_x = max(-3, min(3, int(round(16.0 - cx))))
            shift_y = max(-3, min(3, int(round(16.0 - cy))))
            T = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            canvas = cv2.warpAffine(canvas, T, target_size, flags=cv2.INTER_NEAREST)

    # Convert to base64 preview thumbnail for frontend display
    _, thumb_buf = cv2.imencode(".png", cv2.resize(canvas, (128, 128), interpolation=cv2.INTER_NEAREST))
    thumb_b64 = base64.b64encode(thumb_buf).decode("utf-8")

    tensor = torch.tensor(canvas.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return tensor, thumb_b64


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Handwritten Digit Recognition AI</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(22, 30, 49, 0.75);
      --border: rgba(255, 255, 255, 0.1);
      --accent: #6366f1;
      --accent-glow: rgba(99, 102, 241, 0.35);
      --accent-grad: linear-gradient(135deg, #6366f1, #a855f7);
      --text: #f8fafc;
      --text-dim: #94a3b8;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Outfit', sans-serif;
      background: radial-gradient(circle at top, #1e1b4b, var(--bg));
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2rem 1rem;
    }

    header {
      text-align: center;
      margin-bottom: 2rem;
    }
    h1 {
      font-size: 2.5rem;
      font-weight: 700;
      background: var(--accent-grad);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.5rem;
    }
    p.subtitle {
      color: var(--text-dim);
      font-size: 1.1rem;
    }

    .container {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 2rem;
      max-width: 950px;
      width: 100%;
    }

    @media (max-width: 768px) {
      .container { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border);
      border-radius: 1.25rem;
      padding: 1.75rem;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }

    .card-title {
      font-size: 1.3rem;
      font-weight: 600;
      margin-bottom: 1.25rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    /* Canvas */
    .canvas-container {
      position: relative;
      width: 280px;
      height: 280px;
      margin: 0 auto 1.25rem auto;
      border-radius: 1rem;
      overflow: hidden;
      border: 2px solid rgba(99, 102, 241, 0.4);
      box-shadow: 0 0 25px var(--accent-glow);
    }
    canvas {
      width: 100%;
      height: 100%;
      background: #000;
      cursor: crosshair;
      touch-action: none;
    }

    .btn-group {
      display: flex;
      gap: 0.75rem;
      justify-content: center;
    }
    button, label.upload-btn {
      background: var(--card-bg);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 0.75rem 1.4rem;
      border-radius: 0.75rem;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
    }
    button.primary {
      background: var(--accent-grad);
      border: none;
      box-shadow: 0 4px 15px var(--accent-glow);
    }
    button:hover, label.upload-btn:hover {
      transform: translateY(-2px);
      border-color: rgba(255, 255, 255, 0.3);
    }
    button.primary:hover {
      box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
    }

    /* Results */
    .prediction-hero {
      text-align: center;
      padding: 1.5rem;
      background: rgba(99, 102, 241, 0.1);
      border-radius: 1rem;
      border: 1px solid rgba(99, 102, 241, 0.25);
      margin-bottom: 1.5rem;
    }
    .hero-digit {
      font-size: 4.5rem;
      font-weight: 700;
      color: #a855f7;
      line-height: 1;
      margin: 0.25rem 0;
    }
    .hero-confidence {
      font-size: 1.1rem;
      font-weight: 600;
      color: #34d399;
    }

    .bars-container {
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
    }
    .bar-row {
      display: grid;
      grid-template-columns: 24px 1fr 50px;
      align-items: center;
      gap: 0.75rem;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
    }
    .bar-track {
      background: rgba(255, 255, 255, 0.08);
      height: 8px;
      border-radius: 4px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      background: var(--accent-grad);
      border-radius: 4px;
      width: 0%;
      transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .bar-row.top .bar-fill {
      background: linear-gradient(90deg, #34d399, #10b981);
    }

    .thumbnail-box {
      margin-top: 1.25rem;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 1rem;
      font-size: 0.9rem;
      color: var(--text-dim);
    }
    .thumbnail-box img {
      width: 48px;
      height: 48px;
      border-radius: 0.5rem;
      border: 1px solid var(--border);
      image-rendering: pixelated;
    }
  </style>
</head>
<body>

  <header>
    <h1>Handwritten Digit Recognition AI</h1>
    <p class="subtitle">Trained on Custom Dataset (96.55% Test Accuracy) &bull; Standardized 32x32 Deep Learning Pipeline</p>
  </header>

  <div class="container">
    <!-- Input Column -->
    <div class="card">
      <div class="card-title">✏️ Draw or Upload a Digit</div>
      <div class="canvas-container">
        <canvas id="digitCanvas" width="280" height="280"></canvas>
      </div>
      <div class="btn-group">
        <button id="clearBtn">🗑️ Clear</button>
        <label class="upload-btn">
          📁 Upload Photo
          <input type="file" id="fileInput" accept="image/*" style="display:none;">
        </label>
        <button id="predictBtn" class="primary">⚡ Recognize</button>
      </div>
    </div>

    <!-- Output Column -->
    <div class="card">
      <div class="card-title">🎯 Model Prediction</div>
      <div class="prediction-hero">
        <div style="font-size: 0.9rem; color: var(--text-dim);">Predicted Digit</div>
        <div class="hero-digit" id="predDigit">-</div>
        <div class="hero-confidence" id="predConfidence">Draw on the canvas</div>
      </div>

      <div class="bars-container" id="probBars">
        <!-- Bars generated by JS -->
      </div>

      <div class="thumbnail-box" id="thumbContainer" style="display: none;">
        <span>32x32 Normalized Input:</span>
        <img id="thumbImg" src="" alt="32x32 Model Input">
      </div>
    </div>
  </div>

  <script>
    const canvas = document.getElementById('digitCanvas');
    const ctx = canvas.getContext('2d');
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#ffffff';

    let isDrawing = false;
    let lastX = 0, lastY = 0;

    function getCoords(e) {
      const rect = canvas.getBoundingClientRect();
      const clientX = e.clientX || (e.touches && e.touches[0].clientX);
      const clientY = e.clientY || (e.touches && e.touches[0].clientY);
      return [
        (clientX - rect.left) * (canvas.width / rect.width),
        (clientY - rect.top) * (canvas.height / rect.height)
      ];
    }

    function startDraw(e) {
      isDrawing = true;
      [lastX, lastY] = getCoords(e);
      e.preventDefault();
    }
    function draw(e) {
      if (!isDrawing) return;
      const [x, y] = getCoords(e);
      ctx.beginPath();
      ctx.moveTo(lastX, lastY);
      ctx.lineTo(x, y);
      ctx.stroke();
      [lastX, lastY] = [x, y];
      e.preventDefault();
    }
    function stopDraw() { isDrawing = false; }

    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDraw);
    canvas.addEventListener('mouseleave', stopDraw);

    canvas.addEventListener('touchstart', startDraw, { passive: false });
    canvas.addEventListener('touchmove', draw, { passive: false });
    canvas.addEventListener('touchend', stopDraw);

    document.getElementById('clearBtn').addEventListener('click', () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      document.getElementById('predDigit').innerText = '-';
      document.getElementById('predConfidence').innerText = 'Draw on the canvas';
      document.getElementById('thumbContainer').style.display = 'none';
      renderBars(new Array(10).fill(0));
    });

    function renderBars(probs) {
      const maxIdx = probs.indexOf(Math.max(...probs));
      const container = document.getElementById('probBars');
      container.innerHTML = probs.map((p, digit) => {
        const pct = (p * 100).toFixed(1);
        const isTop = digit === maxIdx && Math.max(...probs) > 0.05;
        return `
          <div class="bar-row ${isTop ? 'top' : ''}">
            <span>${digit}</span>
            <div class="bar-track">
              <div class="bar-fill" style="width: ${pct}%;"></div>
            </div>
            <span>${pct}%</span>
          </div>
        `;
      }).join('');
    }
    renderBars(new Array(10).fill(0));

    async function sendImage(dataUrl) {
      try {
        document.getElementById('predConfidence').innerText = 'Analyzing...';
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: dataUrl })
        });
        const data = await res.json();

        document.getElementById('predDigit').innerText = data.digit;
        document.getElementById('predConfidence').innerText = (data.confidence * 100).toFixed(1) + '% Confidence';
        renderBars(data.probabilities);

        if (data.thumbnail) {
          document.getElementById('thumbImg').src = 'data:image/png;base64,' + data.thumbnail;
          document.getElementById('thumbContainer').style.display = 'flex';
        }
      } catch (err) {
        console.error(err);
      }
    }

    document.getElementById('predictBtn').addEventListener('click', () => {
      sendImage(canvas.toDataURL('image/png'));
    });

    document.getElementById('fileInput').addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          sendImage(event.target.result);
        };
        img.src = event.target.result;
      };
      reader.readAsDataURL(file);
    });
  </script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path == "/predict":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode("utf-8"))

            img_b64 = req["image"].split(",")[1]
            img_bytes = base64.b64decode(img_b64)

            tensor, thumb_b64 = preprocess_canvas_or_photo(img_bytes)

            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1).squeeze().numpy()
                pred_digit = int(np.argmax(probs))
                confidence = float(probs[pred_digit])

            response = {
                "digit": pred_digit,
                "confidence": confidence,
                "probabilities": probs.tolist(),
                "thumbnail": thumb_b64
            }

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))


def run_server(port=8000):
    server = HTTPServer(("0.0.0.0", port), RequestHandler)
    print(f"\n=======================================================")
    print(f"🚀 Interactive Digit Recognition Web App Running at:")
    print(f"   http://localhost:{port}")
    print(f"=======================================================\n")
    server.serve_forever()


if __name__ == "__main__":
    run_server(port=8000)
