using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Numerics;
using System.Reflection;

namespace AdaptiveFilter
{
    public class WebInterface : IDisposable
    {
        private readonly HttpListener _listener;
        private readonly ConcurrentDictionary<string, WebSocket> _sockets = new();
        private readonly CancellationTokenSource _cts = new();
        private Task? _listenTask;

        public WebInterface(int port)
        {
            _listener = new HttpListener();
            _listener.Prefixes.Add($"http://localhost:{port}/");
        }

        public void Start()
        {
            try
            {
                _listener.Start();
                _listenTask = Task.Run(ListenLoop);
            }
            catch (Exception ex)
            {
                // Log error
                Console.WriteLine($"Failed to start listener: {ex.Message}");
            }
        }

        private async Task ListenLoop()
        {
            while (!_cts.Token.IsCancellationRequested && _listener.IsListening)
            {
                try
                {
                    var context = await _listener.GetContextAsync();
                    if (context.Request.IsWebSocketRequest)
                    {
                        ProcessWebSocketRequest(context);
                    }
                    else
                    {
                        ProcessHttpRequest(context);
                    }
                }
                catch (HttpListenerException)
                {
                    // Listener stopped
                    break;
                }
                catch (Exception)
                {
                    // Ignore other errors
                }
            }
        }

        private async void ProcessWebSocketRequest(HttpListenerContext context)
        {
            try
            {
                var wsContext = await context.AcceptWebSocketAsync(null);
                var ws = wsContext.WebSocket;
                var id = Guid.NewGuid().ToString();
                _sockets.TryAdd(id, ws);

                await Echo(ws);
                
                _sockets.TryRemove(id, out _);
                ws.Dispose();
            }
            catch
            {
                context.Response.StatusCode = 500;
                context.Response.Close();
            }
        }

        private void ProcessHttpRequest(HttpListenerContext context)
        {
            string responseString = "";
            string contentType = "text/html";

            if (context.Request.Url?.AbsolutePath == "/" || context.Request.Url?.AbsolutePath == "/index.html")
            {
                // Load index.html from embedded resource or file
                // For simplicity in this environment, we'll read the file directly if possible, 
                // or embed a simple string if we can't rely on file path.
                // Since we know the path structure:
                string path = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? "", "wwwroot", "index.html");
                
                // Fallback if file not found (e.g. not copied to bin)
                // We will try to read from the source location if debugging, but best to embed.
                // For now, let's try to read the file we created.
                // If it fails, we serve a simple error page.
                
                // Actually, let's just hardcode the HTML here to avoid file copy issues in OTD plugin folder.
                // It makes the plugin self-contained.
                responseString = GetEmbeddedHtml();
            }
            else
            {
                context.Response.StatusCode = 404;
                context.Response.Close();
                return;
            }

            byte[] buffer = Encoding.UTF8.GetBytes(responseString);
            context.Response.ContentLength64 = buffer.Length;
            context.Response.ContentType = contentType;
            context.Response.OutputStream.Write(buffer, 0, buffer.Length);
            context.Response.OutputStream.Close();
        }

        private string GetEmbeddedHtml()
        {
            return @"<!DOCTYPE html>
<html lang=""en"">
<head>
    <meta charset=""UTF-8"">
    <meta name=""viewport"" content=""width=device-width, initial-scale=1.0"">
    <title>Adaptive Filter Visualization</title>
    <style>
        body { background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; height: 100vh; overflow: hidden; box-sizing: border-box; }
        h1 { text-align: center; margin: 0 0 20px 0; }
        #stats { display: flex; gap: 15px; justify-content: center; margin-bottom: 20px; flex-wrap: wrap; }
        .stat-box { background: #252525; padding: 12px 15px; border-radius: 8px; min-width: 110px; text-align: center; }
        .stat-value { font-size: 20px; font-weight: bold; color: #4caf50; }
        .stat-label { font-size: 11px; color: #aaa; margin-top: 4px; }
        #main-container { display: flex; gap: 20px; height: calc(100vh - 160px); }
        #left-panel { flex: 1; min-width: 0; }
        #right-panel { width: 400px; }
        #canvas-container { position: relative; height: 100%; border: 1px solid #333; background-color: #1e1e1e; box-shadow: 0 0 20px rgba(0,0,0,0.5); border-radius: 4px; overflow: hidden; }
        canvas { display: block; }
        .legend { position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.7); padding: 10px; border-radius: 4px; font-size: 12px; }
        .legend-item { display: flex; align-items: center; margin-bottom: 5px; }
        .legend-item:last-child { margin-bottom: 0; }
        .color-box { width: 12px; height: 12px; margin-right: 8px; }
        #weights-container { height: 100%; background: #1e1e1e; border: 1px solid #333; padding: 15px; border-radius: 4px; box-sizing: border-box; display: flex; flex-direction: column; }
        #weights-container h3 { margin: 0 0 10px 0; font-size: 16px; }
        #network-viz { flex: 1; background: #252525; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>Adaptive AI Filter (Neural Network)</h1>
    <div id=""stats"">
        <div class=""stat-box""><div class=""stat-value"" id=""latency"">0 ms</div><div class=""stat-label"">Prediction Offset</div></div>
        <div class=""stat-box""><div class=""stat-value"" id=""rate"">0 Hz</div><div class=""stat-label"">Output Rate</div></div>
        <div class=""stat-box""><div class=""stat-value"" id=""accuracy"">0.00</div><div class=""stat-label"">Accuracy (mm)</div></div>
        <div class=""stat-box""><div class=""stat-value"" id=""iterations"">0</div><div class=""stat-label"">Training Iterations</div></div>
    </div>
    <div id=""main-container"">
        <div id=""left-panel"">
            <div id=""canvas-container"">
                <canvas id=""visualizer""></canvas>
                <div class=""legend"">
                    <div class=""legend-item""><div class=""color-box"" style=""background: #00bcd4""></div>Raw Input</div>
                    <div class=""legend-item""><div class=""color-box"" style=""background: #ff4081""></div>AI Prediction</div>
                </div>
            </div>
        </div>
        <div id=""right-panel"">
            <div id=""weights-container"">
                <h3>Neural Network Structure</h3>
                <canvas id=""network-viz""></canvas>
            </div>
        </div>
    </div>
    <script>
        const visualizer = document.getElementById('visualizer');
        const vCtx = visualizer.getContext('2d');
        const networkCanvas = document.getElementById('network-viz');
        const nCtx = networkCanvas.getContext('2d');
        const rateEl = document.getElementById('rate');
        const accEl = document.getElementById('accuracy');
        const iterEl = document.getElementById('iterations');
        let rawPoints = [], predPoints = [];
        const maxPoints = 100;
        let avgAccuracy = 0;

        function resizeCanvases() {
            const container = document.getElementById('canvas-container');
            visualizer.width = container.clientWidth;
            visualizer.height = container.clientHeight;
            
            networkCanvas.width = networkCanvas.offsetWidth;
            networkCanvas.height = networkCanvas.offsetHeight;
        }
        
        // Initial resize
        resizeCanvases();
        
        // Debounced resize handler
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(resizeCanvases, 100);
        });

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                const point = { x: data.x, y: data.y, t: data.t };
                if (data.p) {
                    predPoints.push(point);
                    if (predPoints.length > maxPoints) predPoints.shift();
                } else {
                    rawPoints.push(point);
                    if (rawPoints.length > maxPoints) rawPoints.shift();
                }
                
                if (data.a) {
                    avgAccuracy = avgAccuracy * 0.9 + data.a * 0.1;
                    accEl.textContent = avgAccuracy.toFixed(2);
                }
                
                if (data.r) {
                    rateEl.textContent = `${data.r.toFixed(0)} Hz`;
                }

                if (data.it !== undefined) {
                    iterEl.textContent = data.it.toLocaleString();
                }

                if (data.w && data.ls) {
                    requestAnimationFrame(() => drawNetwork(data.w, data.ls));
                }
            };
            ws.onclose = () => setTimeout(connect, 1000);
        }

        let lastNetworkDraw = 0;
        function drawNetwork(weights, layerSizes) {
            // Throttle network drawing to prevent freezing
            const now = Date.now();
            if (now - lastNetworkDraw < 100) return; // Max 10 FPS for network viz
            lastNetworkDraw = now;
            
            const w = networkCanvas.width;
            const h = networkCanvas.height;
            nCtx.clearRect(0, 0, w, h);

            const layerSpacing = w / (layerSizes.length + 1);
            const neuronRadius = 6;

            let neurons = [];
            layerSizes.forEach((count, layerIdx) => {
                let layer = [];
                const ySpacing = h / (count + 1);
                for (let i = 0; i < count; i++) {
                    layer.push({
                        x: layerSpacing * (layerIdx + 1),
                        y: ySpacing * (i + 1)
                    });
                }
                neurons.push(layer);
            });

            let weightIdx = 0;
            for (let l = 0; l < layerSizes.length - 1; l++) {
                for (let j = 0; j < layerSizes[l + 1]; j++) {
                    for (let i = 0; i < layerSizes[l]; i++) {
                        const weight = weights[weightIdx++];
                        const from = neurons[l][i];
                        const to = neurons[l + 1][j];
                        
                        const intensity = Math.min(1, Math.abs(weight) * 2);
                        const r = weight > 0 ? 255 : 0;
                        const b = weight < 0 ? 255 : 0;
                        
                        nCtx.strokeStyle = `rgba(${r}, 0, ${b}, ${intensity * 0.3})`;
                        nCtx.lineWidth = Math.max(0.5, intensity * 1.5);
                        nCtx.beginPath();
                        nCtx.moveTo(from.x, from.y);
                        nCtx.lineTo(to.x, to.y);
                        nCtx.stroke();
                    }
                }
            }

            neurons.forEach((layer, layerIdx) => {
                layer.forEach(neuron => {
                    nCtx.fillStyle = layerIdx === 0 ? '#00bcd4' : (layerIdx === neurons.length - 1 ? '#ff4081' : '#4caf50');
                    nCtx.beginPath();
                    nCtx.arc(neuron.x, neuron.y, neuronRadius, 0, Math.PI * 2);
                    nCtx.fill();
                    nCtx.strokeStyle = '#fff';
                    nCtx.lineWidth = 2;
                    nCtx.stroke();
                });
            });
        }

        function draw() {
            vCtx.clearRect(0, 0, visualizer.width, visualizer.height);
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            const allPoints = [...rawPoints, ...predPoints];
            if (allPoints.length === 0) { requestAnimationFrame(draw); return; }
            allPoints.forEach(p => {
                if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
                if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
            });
            const padding = 20;
            const rangeX = Math.max(maxX - minX, 10);
            const rangeY = Math.max(maxY - minY, 10);
            const scale = Math.min((visualizer.width - padding * 2) / rangeX, (visualizer.height - padding * 2) / rangeY);
            const offsetX = (visualizer.width - rangeX * scale) / 2 - minX * scale;
            const offsetY = (visualizer.height - rangeY * scale) / 2 - minY * scale;
            function transform(p) { return { x: p.x * scale + offsetX, y: p.y * scale + offsetY }; }
            
            vCtx.beginPath(); vCtx.strokeStyle = '#00bcd4'; vCtx.lineWidth = 2;
            rawPoints.forEach((p, i) => { const t = transform(p); if (i === 0) vCtx.moveTo(t.x, t.y); else vCtx.lineTo(t.x, t.y); });
            vCtx.stroke();
            
            vCtx.fillStyle = '#00bcd4';
            rawPoints.forEach(p => { const t = transform(p); vCtx.beginPath(); vCtx.arc(t.x, t.y, 3, 0, Math.PI * 2); vCtx.fill(); });

            vCtx.beginPath(); vCtx.strokeStyle = '#ff4081'; vCtx.lineWidth = 2;
            predPoints.forEach((p, i) => { const t = transform(p); if (i === 0) vCtx.moveTo(t.x, t.y); else vCtx.lineTo(t.x, t.y); });
            vCtx.stroke();
            requestAnimationFrame(draw);
        }
        
        connect(); draw();
    </script>
</body>
</html>";
        }

        public void BroadcastData(Vector2 pos, double time, bool isPrediction, float accuracy = 0, double[]? weights = null, float rate = 0, int[]? layerSizes = null, int iterations = 0)
        {
            if (_sockets.IsEmpty) return;

            // Manual JSON formatting to avoid allocation overhead
            // {"x":123.45,"y":678.90,"t":123456.78,"p":true,"a":0.12,"r":1000.0,"w":[...],"ls":[10,16,16,2],"it":12345}
            StringBuilder sb = new StringBuilder();
            sb.AppendFormat(System.Globalization.CultureInfo.InvariantCulture, 
                "{{\"x\":{0:F2},\"y\":{1:F2},\"t\":{2:F2},\"p\":{3},\"a\":{4:F4},\"r\":{5:F1},\"it\":{6}", 
                pos.X, pos.Y, time, isPrediction ? "true" : "false", accuracy, rate, iterations);

            if (weights != null)
            {
                sb.Append(",\"w\":[");
                for (int i = 0; i < weights.Length; i++)
                {
                    sb.Append(weights[i].ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
                    if (i < weights.Length - 1) sb.Append(",");
                }
                sb.Append("]");
            }

            if (layerSizes != null)
            {
                sb.Append(",\"ls\":[");
                for (int i = 0; i < layerSizes.Length; i++)
                {
                    sb.Append(layerSizes[i]);
                    if (i < layerSizes.Length - 1) sb.Append(",");
                }
                sb.Append("]");
            }

            sb.Append("}");

            string json = sb.ToString();
            var bytes = Encoding.UTF8.GetBytes(json);
            var segment = new ArraySegment<byte>(bytes);

            foreach (var socket in _sockets.Values)
            {
                if (socket.State == WebSocketState.Open)
                {
                    // Fire and forget - don't await result to block the loop
                    try
                    {
                        socket.SendAsync(segment, WebSocketMessageType.Text, true, CancellationToken.None);
                    }
                    catch { /* Ignore */ }
                }
            }
        }

        private async Task Echo(WebSocket webSocket)
        {
            var buffer = new byte[1024 * 4];
            while (webSocket.State == WebSocketState.Open)
            {
                var result = await webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);
                if (result.MessageType == WebSocketMessageType.Close)
                {
                    await webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, string.Empty, CancellationToken.None);
                }
            }
        }

        public void Dispose()
        {
            _cts.Cancel();
            _listener.Stop();
            _listener.Close();
        }
    }
}
