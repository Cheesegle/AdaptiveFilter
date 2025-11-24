using System;
using System.Numerics;
using System.Collections.Generic;
using System.Linq;

namespace AdaptiveFilter
{
    public class PredictionCore
    {
        private readonly int _capacity;
        private readonly Queue<TimeSeriesPoint> _points;
        private NeuralNetwork _nn;
        private readonly OneEuroFilter _filterX;
        private readonly OneEuroFilter _filterY;
        
        public bool IsReady => _points.Count >= _capacity;
        public int Complexity { get; set; } = 2; 
        public int[] LayerSizes => _nn.Layers;
        public int TrainingIterations { get; private set; } = 0; 

        public double LearningRate { get; set; } = 0.01;
        
        private int _hiddenSize = 16;
        public int HiddenLayerSize 
        {
            get => _hiddenSize;
            set
            {
                if (_hiddenSize != value)
                {
                    _hiddenSize = value;
                    RebuildNetwork();
                }
            }
        }

        private int _hiddenCount = 2;
        public int HiddenLayerCount
        {
            get => _hiddenCount;
            set
            {
                if (_hiddenCount != value)
                {
                    _hiddenCount = Math.Max(1, value);
                    RebuildNetwork();
                }
            }
        }

        public PredictionCore(int capacity = 5)
        {
            _capacity = capacity;
            _points = new Queue<TimeSeriesPoint>(capacity);
            
            RebuildNetwork();
            
            _filterX = new OneEuroFilter(2.0, 0.001);
            _filterY = new OneEuroFilter(2.0, 0.001);
        }

        private void RebuildNetwork()
        {
            // Input: 10, Output: 2
            // Hidden layers: [size, size, ...] (count times)
            int[] layers = new int[_hiddenCount + 2];
            layers[0] = 10;
            for(int i=0; i<_hiddenCount; i++)
            {
                layers[i+1] = _hiddenSize;
            }
            layers[layers.Length - 1] = 2;
            
            _nn = new NeuralNetwork(layers);
        }

        public void Add(Vector2 point, double time)
        {
            if (_points.Count >= _capacity)
            {
                _points.Dequeue();
            }
            _points.Enqueue(new TimeSeriesPoint(point, time));

            if (IsReady)
            {
                Train();
            }
        }

        private void Train()
        {
            var points = _points.ToArray();
            int numDeltas = _points.Count - 1;
            if (numDeltas < 2) return;

            List<Vector2> deltas = new List<Vector2>();
            for (int i = 0; i < points.Length - 1; i++)
            {
                deltas.Add(points[i+1].Point - points[i].Point);
            }
            
            if (deltas.Count < 6) return;
            
            var trainingDeltas = deltas.Skip(deltas.Count - 6).ToArray();
            
            double[] nnInputs = new double[10];
            for(int i=0; i<5; i++)
            {
                nnInputs[i*2] = trainingDeltas[i].X / 100.0;
                nnInputs[i*2+1] = trainingDeltas[i].Y / 100.0;
            }
            
            double[] targets = new double[2];
            targets[0] = trainingDeltas[5].X / 100.0;
            targets[1] = trainingDeltas[5].Y / 100.0;
            
            _nn.BackPropagate(nnInputs, targets, LearningRate);
            TrainingIterations++;
        }

        public Vector2 Predict(double targetTime, float lookahead = 1.0f)
        {
            if (!IsReady || _points.Count < 2) return _points.LastOrDefault().Point;

            var points = _points.ToArray();
            var lastPoint = points.Last();
            
            Vector2 predictedPos;

            List<Vector2> deltas = new List<Vector2>();
            for (int i = 0; i < points.Length - 1; i++)
            {
                deltas.Add(points[i+1].Point - points[i].Point);
            }
            
            if (deltas.Count < 5) return lastPoint.Point;
            
            var inputDeltas = deltas.Skip(deltas.Count - 5).ToArray();
            double[] nnInputs = new double[10];
            for(int i=0; i<5; i++)
            {
                nnInputs[i*2] = inputDeltas[i].X / 100.0;
                nnInputs[i*2+1] = inputDeltas[i].Y / 100.0;
            }
            
            var output = _nn.FeedForward(nnInputs);
            
            // Safety check: validate output
            if (double.IsNaN(output[0]) || double.IsNaN(output[1]) || 
                double.IsInfinity(output[0]) || double.IsInfinity(output[1]))
            {
                // Network diverged - use last known position
                return lastPoint.Point;
            }
            
            float predDeltaX = (float)(output[0] * 100.0);
            float predDeltaY = (float)(output[1] * 100.0);
            
            predDeltaX *= lookahead;
            predDeltaY *= lookahead;
            
            predictedPos = lastPoint.Point + new Vector2(predDeltaX, predDeltaY);
            
            double smoothedX = _filterX.Filter(predictedPos.X, targetTime);
            double smoothedY = _filterY.Filter(predictedPos.Y, targetTime);
            
            return new Vector2((float)smoothedX, (float)smoothedY);
        }

        public double[] GetModelWeights()
        {
            return _nn.GetWeights();
        }

        public struct TimeSeriesPoint
        {
            public Vector2 Point;
            public double Time;

            public TimeSeriesPoint(Vector2 point, double time)
            {
                Point = point;
                Time = time;
            }
        }
    }
}
