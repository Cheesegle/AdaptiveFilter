using System;
using System.Diagnostics;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;
using OpenTabletDriver.Plugin;
using OpenTabletDriver.Plugin.Attributes;
using OpenTabletDriver.Plugin.Output;
using OpenTabletDriver.Plugin.Tablet;

namespace AdaptiveFilter
{
    [PluginName("AdaptiveFilter")]
    public class AdaptiveFilter : IPositionedPipelineElement<IDeviceReport>, IDisposable
    {
        // Interface members required by OpenTabletDriver.Plugin
        public event Action<IDeviceReport>? Emit;
        public PipelinePosition Position => PipelinePosition.PostTransform;

        private PredictionCore _core = new PredictionCore();
        [Property("Prediction Offset"), Unit("ms"), DefaultPropertyValue(0f)]
        public float PredictionOffset { get; set; }
        private readonly Stopwatch _timer = new Stopwatch();
        private WebInterface? _webInterface;
        private CancellationTokenSource? _cts;
        private Task? _upsampleTask;
        private IDeviceReport? _lastReport;
        private object _lock = new object();
        private double _lastWebUpdate = 0;
        private double _lastConsumeTime = 0;
        private double _lastEmitTime = 0;
        private readonly AntiChatterFilter _antiChatter = new AntiChatterFilter();

        // Accuracy Stats
        private Vector2 _lastPredictedPosForAccuracy;
        private double _lastAccuracyCheckTime;
        private float _currentAccuracy;

        [Property("Dynamic Lookahead"), DefaultPropertyValue(false)]
        public bool UseDynamicLookahead { get; set; } = false;

        [Property("Lookahead"), DefaultPropertyValue(1.0f)]
        public float Lookahead { get; set; } = 1.0f;

        [Property("Use Hybrid Mode"), DefaultPropertyValue(false)]
        public bool UseHybridMode { get; set; } = false;

        [Property("Learning Rate"), DefaultPropertyValue(0.01f)]
        public float LearningRate { get => (float)_core.LearningRate; set => _core.LearningRate = value; }

        [Property("Hidden Layer Size"), DefaultPropertyValue(16)]
        public int HiddenLayerSize { get => _core.HiddenLayerSize; set => _core.HiddenLayerSize = value; }

        [Property("Hidden Layer Count"), DefaultPropertyValue(2)]
        public int HiddenLayerCount { get => _core.HiddenLayerCount; set => _core.HiddenLayerCount = value; }

        [Property("Use Absolute Position"), DefaultPropertyValue(false)]
        public bool UseAbsolutePosition { get => _core.UseAbsolutePosition; set => _core.UseAbsolutePosition = value; }

        [Property("Use Time Delta"), DefaultPropertyValue(false)]
        public bool UseTimeDelta { get => _core.UseTimeDelta; set => _core.UseTimeDelta = value; }

        [Property("Use Interpolated Training"), DefaultPropertyValue(false)]
        public bool UseInterpolatedTraining { get => _core.UseInterpolatedTraining; set => _core.UseInterpolatedTraining = value; }

        [Property("Use Predicted Input"), DefaultPropertyValue(false)]
        public bool UsePredictedInput { get => _core.UsePredictedInput; set => _core.UsePredictedInput = value; }

        [Property("Target Rate"), Unit("Hz"), DefaultPropertyValue(1000f)]
        public float TargetRate { get; set; } = 1000f;

        [Property("Samples"), DefaultPropertyValue(5)]
        public int Samples 
        { 
            get => _core.Complexity + 1; 
            set => _core = new PredictionCore(Math.Max(2, value)); 
        }

        [Property("Anti-Chatter Strength"), Unit("mm"), DefaultPropertyValue(1.0f)]
        public float AntiChatterStrength { get => _antiChatter.Strength; set => _antiChatter.Strength = value; }

        [Property("Web UI Port"), DefaultPropertyValue(5000)]
        public int WebPort { get; set; } = 5000;

        [Property("Bypass OTD Output"), DefaultPropertyValue(false)]
        public bool BypassOTD { get; set; } = false;

        public AdaptiveFilter()
        {
            _timer.Start();
            InitializeWebUI();
            StartUpsampling();
        }

        private void InitializeWebUI()
        {
            try 
            {
                _webInterface = new WebInterface(WebPort);
                _webInterface.Start();
            }
            catch (Exception ex)
            {
                Log.Write("AdaptiveFilter", $"Failed to start Web UI: {ex.Message}", LogLevel.Error);
            }
        }

        private void StartUpsampling()
        {
            _cts = new CancellationTokenSource();
            _upsampleTask = Task.Factory.StartNew(() => 
            {
                var token = _cts.Token;
                double nextTick = _timer.Elapsed.TotalMilliseconds;
                double lastWeightBroadcast = 0;
                
                int outputCount = 0;
                double lastRateCheck = 0;
                float currentRate = 0;
                
                Vector2 lastPos = Vector2.Zero;
                double lastPosTime = 0;

                while (!token.IsCancellationRequested)
                {
                    double interval = 1000.0 / Math.Max(1, TargetRate);
                    double now = _timer.Elapsed.TotalMilliseconds;
                    
                    if (now >= nextTick)
                    {
                        nextTick += interval;
                        if (now > nextTick + interval) nextTick = now + interval;

                        bool isIdle = (now - _lastConsumeTime) > 100;
                        bool skipUpsample = UseHybridMode && (now - _lastEmitTime) < interval;

                        if (!isIdle && !skipUpsample && _lastReport != null && _core.IsReady)
                        {
                            try
                            {
                                float currentLookahead = Lookahead;
                                
                                // Dynamic Lookahead Logic
                                if (UseDynamicLookahead)
                                {
                                    // Calculate velocity in mm/ms
                                    // Using last known real position vs current prediction might be noisy
                                    // Let's use the delta from the PredictionCore's last added points if possible, 
                                    // or just track it here.
                                    // Simple approach: Speed based on last 2 predictions (or inputs)
                                    // Since we are upsampling, we can use the speed of the prediction itself
                                    
                                    // Actually, let's use the speed of the *input* signal for stability
                                    // But we don't have easy access to input history here without querying Core.
                                    // Let's use the speed of the output for now.
                                    
                                    // Better: Use a velocity factor. 
                                    // 1.0 at rest.
                                    // Scale up with speed.
                                    // Assuming max speed ~2.0 mm/ms (very fast flick)
                                    // We want to reach ~5.0 lookahead.
                                    // Formula: 1.0 + (Speed * 2.0)
                                    
                                    // Let's calculate speed from the last emitted position
                                    if (lastPosTime > 0 && now > lastPosTime && _lastReport is ITabletReport lastTabletReport)
                                    {
                                        float dist = Vector2.Distance(lastTabletReport.Position, lastPos);
                                        // We need instantaneous velocity.
                                        // Let's just use a heuristic based on the last prediction delta if available, 
                                        // or just use the fixed Lookahead if we can't calculate speed reliably yet.
                                        
                                        // Actually, PredictionCore has the history. Let's trust the user's "5.0" finding.
                                        // Let's try to infer speed from the last few points in the core? 
                                        // No, Core is private.
                                        
                                        // Let's use the distance between the current prediction and the last report position?
                                        // No, that's error.
                                        
                                        // Let's just use the Lookahead value as a "Max" and scale based on a hardcoded velocity curve.
                                        // We'll calculate speed based on the last 2 emitted points.
                                    }
                                    
                                    // SIMPLER: Just use the input delta magnitude from the Core if we could access it.
                                    // Since we can't easily, let's use the difference between current time and last consume time
                                    // to estimate "freshness".
                                    
                                    // actually, let's implement a simple velocity tracker in Consume
                                }
                                
                                // REVISED DYNAMIC LOOKAHEAD:
                                // We'll use a shared _currentVelocity calculated in Consume()
                                if (UseDynamicLookahead)
                                {
                                    // Scale lookahead: 1.0 (base) + Velocity * Factor
                                    // Velocity is in mm/ms. Typical fast flick is 0.5 - 2.0 mm/ms.
                                    // We want to reach ~5.0.
                                    // 1.0 + (Velocity * 4.0)
                                    // Clamp to range [1.0, 6.0]
                                    currentLookahead = 1.0f + (_currentVelocity * 4.0f);
                                    currentLookahead = Math.Clamp(currentLookahead, 1.0f, 6.0f);
                                }

                                var predictedPos = _core.Predict(now + PredictionOffset, currentLookahead);
                                
                                if (float.IsNaN(predictedPos.X) || float.IsNaN(predictedPos.Y) ||
                                    float.IsInfinity(predictedPos.X) || float.IsInfinity(predictedPos.Y))
                                {
                                    continue;
                                }
                            
                                // Record this predicted output for use in future predictions
                                _core.AddPredictedOutput(predictedPos, now + PredictionOffset);
                            
                                _lastPredictedPosForAccuracy = predictedPos;
                                _lastAccuracyCheckTime = now;

                                if (_lastReport is ITabletReport tabletReport)
                                {
                                    lock (_lock)
                                    {
                                        tabletReport.Position = predictedPos;
                                        
                                        if (BypassOTD)
                                        {
                                            InputInjector.MoveMouse(predictedPos);
                                        }
                                        else
                                        {
                                            Emit?.Invoke(tabletReport);
                                        }
                                        
                                        _lastEmitTime = now;
                                    }
                                    outputCount++;
                                    
                                    // Buffer point for web visualization
                                    _predictionBuffer.Add(predictedPos);
                                    
                                    if (now - _lastPredWebUpdate > 16)
                                    {
                                        double[]? weights = null;
                                        int[]? layerSizes = null;
                                        int iterations = _core.TrainingIterations;
                                        if (now - lastWeightBroadcast > 250)
                                        {
                                            weights = _core.GetModelWeights();
                                            layerSizes = _core.LayerSizes;
                                            lastWeightBroadcast = now;
                                        }

                                        // Also send a short predicted trajectory for visualization
                                        Vector2[]? predictedSeq = null;
                                        try
                                        {
                                            predictedSeq = _core.PredictSequence(8, currentLookahead);
                                        }
                                        catch
                                        {
                                            predictedSeq = null;
                                        }

                                        _webInterface?.BroadcastData(predictedPos, now, true, _currentAccuracy, weights, currentRate, layerSizes, iterations, predictedSeq, _predictionBuffer);
                                        _predictionBuffer.Clear();
                                        _lastPredWebUpdate = now;
                                    }
                                }
                            }
                            catch
                            {
                            }
                        }
                    }
                    
                    if (now - lastRateCheck > 500)
                    {
                        currentRate = (float)(outputCount * 1000.0 / (now - lastRateCheck));
                        outputCount = 0;
                        lastRateCheck = now;
                    }
                    
                    if (nextTick - now > 1) Thread.Sleep(0);
                    else Thread.SpinWait(10);
                }
            }, _cts.Token, TaskCreationOptions.LongRunning, TaskScheduler.Default);
        }

        private float _currentVelocity = 0;
        private Vector2 _lastVelocityPos;
        private double _lastVelocityTime;
        // Separate web update timestamps to avoid throttling races between
        // raw input broadcasts and predicted broadcasts.
        private double _lastRawWebUpdate = 0;
        private double _lastPredWebUpdate = 0;
        private List<Vector2> _predictionBuffer = new List<Vector2>();

        public void Consume(IDeviceReport value)
        {
            if (value is ITabletReport report)
            {
                lock (_lock)
                {
                    _lastReport = report;
                    double now = _timer.Elapsed.TotalMilliseconds;
                    
                    // Calculate Velocity (mm/ms)
                    if (_lastVelocityTime > 0 && now > _lastVelocityTime)
                    {
                        float dist = Vector2.Distance(report.Position, _lastVelocityPos);
                        float dt = (float)(now - _lastVelocityTime);
                        if (dt > 0)
                        {
                            float instantVel = dist / dt;
                            // Smooth velocity slightly
                            _currentVelocity = (_currentVelocity * 0.5f) + (instantVel * 0.5f);
                        }
                    }
                    _lastVelocityPos = report.Position;
                    _lastVelocityTime = now;

                    _lastConsumeTime = now;
                    
                    if (Math.Abs(now - _lastAccuracyCheckTime) < 10)
                    {
                        _currentAccuracy = Vector2.Distance(report.Position, _lastPredictedPosForAccuracy);
                    }

                    var filteredPos = _antiChatter.Filter(report.Position);
                    
                    _core.Add(filteredPos, now);
                    
                    if (UseHybridMode)
                    {
                        report.Position = filteredPos;
                        
                        if (BypassOTD)
                        {
                            InputInjector.MoveMouse(filteredPos);
                        }
                        else
                        {
                            Emit?.Invoke(report);
                        }
                        
                        _lastEmitTime = now;
                    }
                    
                    if (now - _lastRawWebUpdate > 16)
                    {
                        _webInterface?.BroadcastData(filteredPos, now, false, _currentAccuracy);
                        _lastRawWebUpdate = now;
                    }
                }
            }
            
            if (!_core.IsReady)
            {
                if (BypassOTD && value is ITabletReport tr)
                {
                    InputInjector.MoveMouse(tr.Position);
                }
                else
                {
                    Emit?.Invoke(value);
                }
            }
        }

        public void Dispose()
        {
            _cts?.Cancel();
            _webInterface?.Dispose();
            _timer.Stop();
        }
    }
}
