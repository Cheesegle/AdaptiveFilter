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
        private readonly OneEuroFilter _preFilterX = new OneEuroFilter(1.0, 0.1);
        private readonly OneEuroFilter _preFilterY = new OneEuroFilter(1.0, 0.1);
        private readonly OneEuroFilter _postFilterX = new OneEuroFilter(1.0, 0.1);
        private readonly OneEuroFilter _postFilterY = new OneEuroFilter(1.0, 0.1);

        // Upsampling
        private bool _useUpsampling = false;
        private float _upsampleHz = 1000f;
        private Thread? _upsampleThread;
        private CancellationTokenSource? _upsampleCts;
        private int _upsampleEmitCount = 0;
        private double _upsampleRateWindow = 0;
        private float _measuredUpsampleRate = 0;
        // Pen-lift detection: if no tablet report arrives for this many ms
        // the pen is assumed to have left proximity (works for all input types).
        private const double PenLiftTimeoutMs = 50.0;

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

        [Property("Use Future-Input Training"), DefaultPropertyValue(false)]
        public bool UseFutureTraining { get => _core.UseFutureTraining; set => _core.UseFutureTraining = value; }

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

        [Property("Pre-Smoothing Min Cutoff"), Unit("Hz"), DefaultPropertyValue(1.0f)]
        public float PreSmoothingMinCutoff 
        { 
            get => (float)_preFilterX.MinCutoff; 
            set { _preFilterX.MinCutoff = value; _preFilterY.MinCutoff = value; } 
        }

        [Property("Pre-Smoothing Beta"), DefaultPropertyValue(0.1f)]
        public float PreSmoothingBeta 
        { 
            get => (float)_preFilterX.Beta; 
            set { _preFilterX.Beta = value; _preFilterY.Beta = value; } 
        }

        [Property("Use Post-Smoothing"), DefaultPropertyValue(false)]
        public bool UsePostSmoothing { get; set; } = false;

        [Property("Post-Smoothing Min Cutoff"), Unit("Hz"), DefaultPropertyValue(1.0f)]
        public float PostSmoothingMinCutoff 
        { 
            get => (float)_postFilterX.MinCutoff; 
            set { _postFilterX.MinCutoff = value; _postFilterY.MinCutoff = value; } 
        }

        [Property("Post-Smoothing Beta"), DefaultPropertyValue(0.1f)]
        public float PostSmoothingBeta 
        { 
            get => (float)_postFilterX.Beta; 
            set { _postFilterX.Beta = value; _postFilterY.Beta = value; } 
        }

        [Property("Web UI Port"), DefaultPropertyValue(5000)]
        public int WebPort { get; set; } = 5000;

        [Property("Bypass OTD Output"), DefaultPropertyValue(false)]
        public bool BypassOTD { get; set; } = false;

        [Property("Use Pressure Input"), DefaultPropertyValue(false)]
        public bool UsePressureInput { get => _core.UsePressureInput; set => _core.UsePressureInput = value; }

        [Property("Use Hover Distance Input"), DefaultPropertyValue(false)]
        public bool UseHoverDistance { get => _core.UseHoverDistance; set => _core.UseHoverDistance = value; }

        [Property("Pressure History Size"), DefaultPropertyValue(5)]
        public int PressureHistorySize { get => _core.PressureHistorySize; set => _core.PressureHistorySize = value; }

        [Property("Use Upsampling"), DefaultPropertyValue(false)]
        public bool UseUpsampling
        {
            get => _useUpsampling;
            set
            {
                if (_useUpsampling != value)
                {
                    _useUpsampling = value;
                    if (value) StartUpsampleThread();
                    else StopUpsampleThread();
                }
            }
        }

        [Property("Upsample Hz"), Unit("Hz"), DefaultPropertyValue(1000f)]
        public float UpsampleHz
        {
            get => _upsampleHz;
            set
            {
                _upsampleHz = Math.Max(1f, value);
                // Restart the thread so it picks up the new interval
                if (_useUpsampling)
                {
                    StopUpsampleThread();
                    StartUpsampleThread();
                }
            }
        }

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

        private void StartUpsampleThread()
        {
            _upsampleCts = new CancellationTokenSource();
            _upsampleThread = new Thread(UpsampleLoop)
            {
                IsBackground = true,
                Name = "AdaptiveFilter-Upsample",
                Priority = ThreadPriority.AboveNormal
            };
            _upsampleThread.Start();
        }

        private void StopUpsampleThread()
        {
            _upsampleCts?.Cancel();
            _upsampleThread?.Join(500);
            _upsampleThread = null;
            _upsampleCts = null;
        }

        private void UpsampleLoop()
        {
            var token = _upsampleCts!.Token;
            var sw = Stopwatch.StartNew();
            double intervalMs = 1000.0 / _upsampleHz;
            double nextTickMs = sw.Elapsed.TotalMilliseconds + intervalMs;

            while (!token.IsCancellationRequested)
            {
                // Busy-spin with a small sleep to approach target Hz
                double remaining = nextTickMs - sw.Elapsed.TotalMilliseconds;
                if (remaining > 1.5)
                    Thread.Sleep(1);
                else
                {
                    while (sw.Elapsed.TotalMilliseconds < nextTickMs)
                        Thread.SpinWait(10);
                }

                if (token.IsCancellationRequested) break;

                nextTickMs += intervalMs;

                lock (_lock)
                {
                    if (_lastReport == null || !_core.IsReady) continue;

                    // If no real report has arrived recently, the pen left proximity — stop emitting
                    double idleMs = _timer.Elapsed.TotalMilliseconds - _lastConsumeTime;
                    if (idleMs > PenLiftTimeoutMs) continue;

                    double now = _timer.Elapsed.TotalMilliseconds;
                    var predictedPos = _core.Predict(now, PredictionSteps, PredictionGain);

                    if (float.IsNaN(predictedPos.X) || float.IsNaN(predictedPos.Y) ||
                        float.IsInfinity(predictedPos.X) || float.IsInfinity(predictedPos.Y))
                        continue;

                    if (UsePostSmoothing)
                    {
                        predictedPos.X = (float)_postFilterX.Filter(predictedPos.X, now);
                        predictedPos.Y = (float)_postFilterY.Filter(predictedPos.Y, now);
                    }

                    // Safely emit with predicted position without corrupting _lastReport for Consume()
                    if (_lastReport is ITabletReport tabletReport)
                    {
                        // Mutate, emit, restore — safe because we hold _lock
                        var origPos = tabletReport.Position;
                        tabletReport.Position = predictedPos;

                        if (BypassOTD)
                            InputInjector.MoveMouse(predictedPos);
                        else
                            Emit?.Invoke(tabletReport);

                        tabletReport.Position = origPos; // restore

                        _lastEmitTime = now;

                        // Track upsample rate
                        _upsampleEmitCount++;
                        if (now - _upsampleRateWindow >= 1000.0)
                        {
                            _measuredUpsampleRate = (float)(_upsampleEmitCount * 1000.0 / Math.Max(1, now - _upsampleRateWindow));
                            _upsampleEmitCount = 0;
                            _upsampleRateWindow = now;
                        }
                    }
                }
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
                        smoothedPos.X = (float)_preFilterX.Filter(smoothedPos.X, now);
                        smoothedPos.Y = (float)_preFilterY.Filter(smoothedPos.Y, now);
                    }
                    
                    var filteredPos = _antiChatter.Filter(smoothedPos);
                    
                    if (Math.Abs(now - _lastAccuracyCheckTime) < 10)
                    {
                        _currentAccuracy = Vector2.Distance(report.Position, _lastPredictedPosForAccuracy);
                    }
                    
                    // Extract pressure (normalized 0-1) and hover distance
                    float pressure = report.Pressure / 32767f;
                    float hover = 0f; // OTD has no standard hover-distance interface;
                                      // extend here if your driver exposes one

                    _core.Add(filteredPos, now, pressure, hover);
                    
                    if (_core.IsReady && !_useUpsampling)
                    {
                        // 1:1 Prediction Mode (only when upsampling is off)
                        var predictedPos = _core.Predict(now, PredictionSteps, PredictionGain);
                        
                        if (!float.IsNaN(predictedPos.X) && !float.IsNaN(predictedPos.Y) &&
                            !float.IsInfinity(predictedPos.X) && !float.IsInfinity(predictedPos.Y))
                        {
                            if (UsePostSmoothing)
                            {
                                predictedPos.X = (float)_postFilterX.Filter(predictedPos.X, now);
                                predictedPos.Y = (float)_postFilterY.Filter(predictedPos.Y, now);
                            }

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
                    else if (!_core.IsReady && _useUpsampling)
                    {
                        // NN not ready yet — pass raw input through while upsampling
                        if (BypassOTD)
                            InputInjector.MoveMouse(report.Position);
                        else
                            Emit?.Invoke(report);
                    }
                    // When upsampling is on and core is ready, the upsample thread handles emit.
                    
                    float displayRate = _useUpsampling ? _measuredUpsampleRate : 
                        (_lastEmitTime > 0 ? (float)(1000.0 / Math.Max(1, now - _lastEmitTime + 1)) : 0);

                    if (now - _lastRawWebUpdate > 16)
                    {
                        _webInterface?.BroadcastData(filteredPos, now, false, _currentAccuracy, 
                            rate: displayRate);
                        _lastRawWebUpdate = now;
                    }
                }
            }
            
            if (!_core.IsReady && !_useUpsampling)
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
            StopUpsampleThread();
            _webInterface?.Dispose();
            _timer.Stop();
        }
    }
}
