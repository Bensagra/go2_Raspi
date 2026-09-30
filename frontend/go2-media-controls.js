/* Reusable browser client. No framework/build step required. See MEDIA_FRONTEND.md. */
(() => {
  "use strict";
  const endpoint = (base, path) => new URL(base.replace(/\/$/, "") + path);

  async function confirmedCommand({ apiBase, token, robotId }, type, payload, timeoutMs = 8000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    // getRandomValues is also available on HTTP LAN dashboards; randomUUID isn't.
    const commandId = "web-" + Array.from(crypto.getRandomValues(new Uint8Array(16)),
      byte => byte.toString(16).padStart(2, "0")).join("");
    const path = `/api/robots/${encodeURIComponent(robotId)}`;
    async function request(suffix, options = {}) {
      const response = await fetch(endpoint(apiBase, path + suffix), {
        ...options, signal: controller.signal,
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      return response.json();
    }
    try {
      await request("/commands", { method: "POST", body: JSON.stringify({
        command_id: commandId, type, payload, ttl_ms: 3000,
      }) });
      while (!controller.signal.aborted) {
        const history = await request("/replay?limit=200");
        const ack = history.acks.find(item => item.command_id === commandId &&
          ["executed", "error", "rejected"].includes(item.status));
        if (ack?.status === "executed") return ack.result;
        if (ack) throw new Error(ack.reason || "El robot rechazó el comando");
        await new Promise(resolve => setTimeout(resolve, 200));
      }
      throw new Error("Sin confirmación del robot; verificá la conexión");
    } catch (error) {
      if (controller.signal.aborted) throw new Error("Sin confirmación del robot; verificá la conexión");
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  // Downsample continuously across render quanta. Send 40 ms PCM16 LE frames.
  const workletSource = `
    class Go2MicProcessor extends AudioWorkletProcessor {
      constructor() { super(); this.phase = 0; this.sum = 0; this.count = 0;
        this.pcm = new DataView(new ArrayBuffer(1280)); this.index = 0; }
      process(inputs) {
        const channels = inputs[0];
        if (!channels?.length) return true;
        for (let i = 0; i < channels[0].length; i++) {
          let value = 0;
          for (const channel of channels) value += channel[i] / channels.length;
          this.sum += value; this.count++; this.phase += 16000;
          if (this.phase >= sampleRate) {
            this.phase -= sampleRate;
            const sample = Math.max(-1, Math.min(1, this.sum / this.count));
            this.pcm.setInt16(this.index * 2, Math.round(sample * (sample < 0 ? 32768 : 32767)), true);
            this.index++; this.sum = 0; this.count = 0;
            if (this.index === 640) {
              this.port.postMessage(this.pcm.buffer, [this.pcm.buffer]);
              this.pcm = new DataView(new ArrayBuffer(1280)); this.index = 0;
            }
          }
        }
        return true;
      }
    }
    registerProcessor('go2-mic', Go2MicProcessor);
  `;

  class TalkClient {
    constructor(onStatus = () => {}) {
      this.onStatus = onStatus;
      this.run = null;
      this.onBlur = () => this.stop();
      this.onVisibility = () => { if (document.hidden) this.stop(); };
      window.addEventListener("blur", this.onBlur);
      window.addEventListener("pagehide", this.onBlur);
      document.addEventListener("visibilitychange", this.onVisibility);
    }
    get active() { return this.run !== null; }
    async start({ apiBase, token, robotId }) {
      if (this.run) return;
      if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
        throw new Error("El micrófono necesita HTTPS o localhost y un navegador con AudioWorklet");
      }
      const run = {};
      this.run = run;
      this.onStatus({ status: "opening" });
      try {
        run.context = new AudioContext();
        // Resume during the initiating user gesture, before waiting on permission.
        await run.context.resume();
        if (this.run !== run) return;
        const stream = await navigator.mediaDevices.getUserMedia({ audio: {
          channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true,
        } });
        if (this.run !== run) { stream.getTracks().forEach(t => t.stop()); return; }
        run.stream = stream;
        stream.getTracks().forEach(t => t.addEventListener("ended", () => {
          if (this.run === run) this.stop();
        }));
        const blobURL = URL.createObjectURL(new Blob([workletSource], { type: "text/javascript" }));
        try { await run.context.audioWorklet.addModule(blobURL); }
        finally { URL.revokeObjectURL(blobURL); }
        if (this.run !== run) return;
        const url = endpoint(apiBase, `/ws/talk/${encodeURIComponent(robotId)}`);
        url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
        url.searchParams.set("token", token);
        run.ws = new WebSocket(url);
        await new Promise((resolve, reject) => {
          run.timer = setTimeout(() => reject(new Error("No respondió el canal de voz")), 8000);
          run.ws.onmessage = event => {
            try {
              const message = JSON.parse(event.data);
              if (message.status === "ready") resolve();
              else if (message.status === "error") reject(new Error(message.error));
              else if (message.status === "stopped") reject(new Error("Canal de voz cerrado"));
            } catch (e) { reject(e); }
          };
          run.ws.onerror = () => reject(new Error("No se pudo abrir el canal de voz"));
          run.ws.onclose = event => reject(new Error(`Canal de voz cerrado (${event.code})`));
        });
        clearTimeout(run.timer);
        if (this.run !== run) return;
        run.ws.onclose = () => { if (this.run === run) this.stop(); };
        run.ws.onerror = () => { if (this.run === run) this.stop("Se perdió el canal de voz"); };
        run.ws.onmessage = event => {
          if (this.run !== run) return;
          const msg = JSON.parse(event.data);
          if (msg.status === "error" || msg.status === "stopped") this.stop(msg.error || "");
        };
        run.source = run.context.createMediaStreamSource(stream);
        run.processor = new AudioWorkletNode(run.context, "go2-mic");
        run.mute = run.context.createGain();
        run.mute.gain.value = 0;
        run.processor.port.onmessage = event => {
          if (this.run !== run || run.ws.readyState !== WebSocket.OPEN) return;
          if (run.ws.bufferedAmount > 6400) { this.stop("Red lenta: se cortó el micrófono"); return; }
          run.ws.send(event.data);
        };
        run.source.connect(run.processor).connect(run.mute).connect(run.context.destination);
        run.started = Date.now();
        run.timer = setTimeout(() => this.stop(), 20000);
        run.ticker = setInterval(() => this.onStatus({ status: "talking",
          remaining: Math.max(0, 20 - Math.floor((Date.now() - run.started) / 1000)) }), 1000);
        this.onStatus({ status: "talking", remaining: 20 });
      } catch (error) {
        if (this.run !== run) return;
        this.stop(error.message);
        throw error;
      }
    }
    stop(error = "") {
      const run = this.run;
      if (!run) return;
      this.run = null;
      clearTimeout(run.timer); clearInterval(run.ticker);
      if (run.processor) run.processor.port.onmessage = null;
      run.source?.disconnect(); run.processor?.disconnect(); run.mute?.disconnect();
      run.stream?.getTracks().forEach(t => t.stop());
      if (run.context) void run.context.close().catch(() => {});
      if (run.ws) {
        if (run.ws.readyState === WebSocket.OPEN) run.ws.send('{"type":"stop"}');
        run.ws.close();
      }
      this.onStatus({ status: error ? "error" : "stopped", error });
    }
    dispose() {
      this.stop();
      window.removeEventListener("blur", this.onBlur);
      window.removeEventListener("pagehide", this.onBlur);
      document.removeEventListener("visibilitychange", this.onVisibility);
    }
  }

  async function captureCanvas(canvas, { robotId, lastFrameAt, maxAgeMs = 3000 }) {
    if (!Number.isFinite(lastFrameAt) || !lastFrameAt || Date.now() - lastFrameAt > maxAgeMs ||
        lastFrameAt - Date.now() > 2000) {
      throw new Error("No hay un cuadro reciente. Activá la cámara y esperá una imagen");
    }
    // Copy synchronously: async encoding must not capture a later video frame.
    const copy = document.createElement("canvas");
    copy.width = canvas.width; copy.height = canvas.height;
    copy.getContext("2d").drawImage(canvas, 0, 0);
    const blob = await new Promise(resolve => copy.toBlob(resolve, "image/png"));
    if (!blob) throw new Error("No se pudo crear la captura");
    const safeId = String(robotId).replace(/[^a-zA-Z0-9_-]/g, "_");
    return { blob, filename: `${safeId}_${new Date(lastFrameAt).toISOString().replace(/[:.]/g, "-")}.png`,
      frameAt: lastFrameAt, width: copy.width, height: copy.height };
  }

  window.Go2Media = { TalkClient, confirmedCommand, captureCanvas };
})();
