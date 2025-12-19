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
        [Property("Prediction Gain"), DefaultPropertyValue(1.0f)]
        public float PredictionGain { get; set; } = 1.0f;
        private readonly Stopwatch _timer = new Stopwatch();
        private WebInterface? _webInterface;
        private IDeviceReport? _lastReport;
        private object _lock = new object();
        private double _lastConsumeTime = 0;
        private double _lastEmitTime = 0;
        private readonly AntiChatterFilter _antiChatter = new AntiChatterFilter();
        private readonly LowPassFilter _preFilterX = new LowPassFilter();
        private readonly LowPassFilter _preFilterY = new LowPassFilter();

        // Accuracy Stats
        private Vector2 _lastPredictedPosForAccuracy;
        private double _lastAccuracyCheckTime;
        private float _currentAccuracy;

        [Property("Prediction Steps"), DefaultPropertyValue(1)]
        public int PredictionSteps { get; set; } = 1;

        [Property("Learning Rate"), DefaultPropertyValue(0.1f)]
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

        [Property("Samples"), DefaultPropertyValue(10)]
        public int Samples 
        { 
            get => _core.IsReady ? _core.LayerSizes[0] / 2 + 1 : 10; 
            set => _core = new PredictionCore(Math.Max(8, value)); 
        }

        [Property("Anti-Chatter Strength"), Unit("mm"), DefaultPropertyValue(1.0f)]
        public float AntiChatterStrength { get => _antiChatter.Strength; set => _antiChatter.Strength = value; }

        [Property("Use Pre-Smoothing"), DefaultPropertyValue(false)]
        public bool UsePreSmoothing { get; set; } = false;

        [Property("Smoothing Latency"), Unit("ms"), DefaultPropertyValue(2.0f)]
        public float PreSmoothingLatency { get; set; } = 2.0f;

        [Property("Web UI Port"), DefaultPropertyValue(5000)]
        public int WebPort { get; set; } = 5000;

        [Property("Bypass OTD Output"), DefaultPropertyValue(false)]
        public bool BypassOTD { get; set; } = false;

        public AdaptiveFilter()
        {
            _timer.Start();
            InitializeWebUI();
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


        // Web visualization tracking

        private double _lastRawWebUpdate = 0;

        public void Consume(IDeviceReport value)
        {
            if (value is ITabletReport report)
            {
                lock (_lock)
                {
                    _lastReport = report;
                    double now = _timer.Elapsed.TotalMilliseconds;
                    double dt_consume = now - _lastConsumeTime;
                    _lastConsumeTime = now;

                    Vector2 smoothedPos = report.Position;
                    if (UsePreSmoothing)
                    {
                        double filterDt = (dt_consume <= 0 || dt_consume > 100) ? 1.0 : dt_consume;
                        double alpha = filterDt / (filterDt + PreSmoothingLatency);
                        
                        smoothedPos.X = (float)_preFilterX.Filter(smoothedPos.X, alpha);
                        smoothedPos.Y = (float)_preFilterY.Filter(smoothedPos.Y, alpha);
                    }
                    
                    var filteredPos = _antiChatter.Filter(smoothedPos);
                    
                    if (Math.Abs(now - _lastAccuracyCheckTime) < 10)
                    {
                        _currentAccuracy = Vector2.Distance(report.Position, _lastPredictedPosForAccuracy);
                    }
                    
                    _core.Add(filteredPos, now);
                    
                    if (_core.IsReady)
                    {
                        // 1:1 Prediction Mode
                        var predictedPos = _core.Predict(now, PredictionSteps, PredictionGain);
                        
                        if (!float.IsNaN(predictedPos.X) && !float.IsNaN(predictedPos.Y) &&
                            !float.IsInfinity(predictedPos.X) && !float.IsInfinity(predictedPos.Y))
                        {
                            report.Position = predictedPos;
                            
                            if (BypassOTD)
                            {
                                InputInjector.MoveMouse(predictedPos);
                            }
                            else
                            {
                                Emit?.Invoke(report);
                            }
                            
                            _lastEmitTime = now;
                            _lastPredictedPosForAccuracy = predictedPos;
                            _lastAccuracyCheckTime = now;
                        }
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
            _webInterface?.Dispose();
            _timer.Stop();
        }
    }
}
