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
        private PredictionCore _core = new PredictionCore();
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

        public event Action<IDeviceReport>? Emit;

        public PipelinePosition Position => PipelinePosition.PostTransform;

        [Property("Prediction Offset"), Unit("ms"), DefaultPropertyValue(0f)]
        public float PredictionOffset { get; set; }

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
                double interval = 1000.0 / TargetRate;
                double nextTick = _timer.Elapsed.TotalMilliseconds;
                double lastWeightBroadcast = 0;
                
                int outputCount = 0;
                double lastRateCheck = 0;
                float currentRate = 0;

                while (!token.IsCancellationRequested)
                {
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
                                var predictedPos = _core.Predict(now + PredictionOffset, Lookahead);
                                
                                if (float.IsNaN(predictedPos.X) || float.IsNaN(predictedPos.Y) ||
                                    float.IsInfinity(predictedPos.X) || float.IsInfinity(predictedPos.Y))
                                {
                                    continue;
                                }
                            
                                _lastPredictedPosForAccuracy = predictedPos;
                                _lastAccuracyCheckTime = now;

                                if (_lastReport is ITabletReport tabletReport)
                                {
                                    lock (_lock)
                                    {
                                        tabletReport.Position = predictedPos;
                                        Emit?.Invoke(tabletReport);
                                        _lastEmitTime = now;
                                    }
                                    outputCount++;
                                    
                                    if (now - _lastWebUpdate > 16)
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

                                        _webInterface?.BroadcastData(predictedPos, now, true, _currentAccuracy, weights, currentRate, layerSizes, iterations);
                                        _lastWebUpdate = now;
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

        public void Consume(IDeviceReport value)
        {
            if (value is ITabletReport report)
            {
                lock (_lock)
                {
                    _lastReport = report;
                    double now = _timer.Elapsed.TotalMilliseconds;
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
                        Emit?.Invoke(report);
                        _lastEmitTime = now;
                    }
                    
                    if (now - _lastWebUpdate > 16)
                    {
                        _webInterface?.BroadcastData(filteredPos, now, false, _currentAccuracy);
                        _lastWebUpdate = now;
                    }
                }
            }
            
            if (!_core.IsReady)
            {
                Emit?.Invoke(value);
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
