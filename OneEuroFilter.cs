using System;

namespace AdaptiveFilter
{
    public class OneEuroFilter
    {
        public double MinCutoff { get => _minCutoff; set => _minCutoff = value; }
        public double Beta { get => _beta; set => _beta = value; }

        private double _minCutoff;
        private double _beta;
        private double _dCutoff;
        private LowPassFilter _xFilter;
        private LowPassFilter _dxFilter;
        private double _lastTime;

        public OneEuroFilter(double minCutoff, double beta)
        {
            _minCutoff = minCutoff;
            _beta = beta;
            _dCutoff = 1.0;
            _xFilter = new LowPassFilter();
            _dxFilter = new LowPassFilter();
            _lastTime = -1;
        }

        public double Filter(double value, double timestamp)
        {
            // If first time, return value
            if (_lastTime == -1)
            {
                _lastTime = timestamp;
                _xFilter.Filter(value, 0); // Initialize
                return value;
            }

            double dt = timestamp - _lastTime;
            // Avoid division by zero or negative time
            if (dt <= 0) dt = 0.001; 
            
            _lastTime = timestamp;

            double dx = (value - _xFilter.LastValue) / dt;
            double edx = _dxFilter.Filter(dx, Alpha(dt, _dCutoff));
            double cutoff = _minCutoff + _beta * Math.Abs(edx);
            return _xFilter.Filter(value, Alpha(dt, cutoff));
        }

        private double Alpha(double dt, double cutoff)
        {
            double tau = 1.0 / (2 * Math.PI * cutoff);
            return 1.0 / (1.0 + tau / dt);
        }
    }

    public class LowPassFilter
    {
        public double LastValue { get; private set; }
        private bool _initialized;

        public double Filter(double value, double alpha)
        {
            if (!_initialized)
            {
                LastValue = value;
                _initialized = true;
                return value;
            }
            double result = alpha * value + (1.0 - alpha) * LastValue;
            LastValue = result;
            return result;
        }
    }
}
