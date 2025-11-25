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

        private bool _useAbsolutePosition = false;
        public bool UseAbsolutePosition
        {
            get => _useAbsolutePosition;
            set
            {
                if (_useAbsolutePosition != value)
                {
                    _useAbsolutePosition = value;
                    RebuildNetwork();
                }
            }
        }

        private bool _useTimeDelta = false;
        public bool UseTimeDelta
        {
            get => _useTimeDelta;
            set
            {
                if (_useTimeDelta != value)
                {
                    _useTimeDelta = value;
                    RebuildNetwork();
                }
            }
        }

        private bool _useInterpolatedTraining = false;
        public bool UseInterpolatedTraining
        {
            get => _useInterpolatedTraining;
            set => _useInterpolatedTraining = value;
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
            // Calculate input size based on features
            // Base: 5 deltas × 2 (X,Y) = 10
            // + Absolute positions: 5 positions × 2 (X,Y) = 10
            // + Time deltas: 5 time differences = 5
            int inputSize = 10; // Base deltas
            if (_useAbsolutePosition) inputSize += 10; // Absolute X,Y positions
            if (_useTimeDelta) inputSize += 5; // Time differences

            int[] layers = new int[_hiddenCount + 2];
            layers[0] = inputSize;
            for(int i=0; i<_hiddenCount; i++)
            {
                layers[i+1] = _hiddenSize;
            }
            layers[layers.Length - 1] = 2;
            
            _nn = new NeuralNetwork(layers);
            TrainingIterations = 0; // Reset training count when network changes
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
            var trainingPoints = points.Skip(points.Length - 6).ToArray();
            
            // Train on actual data
            TrainOnSequence(trainingDeltas, trainingPoints);
            
            // Optionally train on interpolated data for the last 1-2 inputs
            if (_useInterpolatedTraining && trainingPoints.Length >= 3)
            {
                // Interpolate between last 2 inputs (indices 4 and 5)
                var interpolated1 = InterpolateTrainingData(trainingPoints, trainingDeltas, 4, 5);
                if (interpolated1 != null)
                {
                    TrainOnSequence(interpolated1.Value.deltas, interpolated1.Value.points);
                }
                
                // Interpolate between inputs 3 and 4
                var interpolated2 = InterpolateTrainingData(trainingPoints, trainingDeltas, 3, 4);
                if (interpolated2 != null)
                {
                    TrainOnSequence(interpolated2.Value.deltas, interpolated2.Value.points);
                }
            }
        }

        private (Vector2[] deltas, TimeSeriesPoint[] points)? InterpolateTrainingData(
            TimeSeriesPoint[] originalPoints, Vector2[] originalDeltas, int idx1, int idx2)
        {
            if (idx1 < 0 || idx2 >= originalPoints.Length) return null;
            
            var p1 = originalPoints[idx1];
            var p2 = originalPoints[idx2];
            
            // Linear interpolation at midpoint in time
            double timeDelta = p2.Time - p1.Time;
            if (timeDelta <= 0) return null;
            
            double midTime = p1.Time + (timeDelta / 2.0);
            float t = 0.5f; // Interpolation factor
            
            Vector2 midPoint = Vector2.Lerp(p1.Point, p2.Point, t);
            var interpolatedPoint = new TimeSeriesPoint(midPoint, midTime);
            
            // Build new sequence with interpolated point
            var newPoints = new TimeSeriesPoint[6];
            var newDeltas = new Vector2[6];
            
            // Copy points before interpolation
            for (int i = 0; i < idx1; i++)
            {
                newPoints[i] = originalPoints[i];
            }
            
            // Insert interpolated point
            newPoints[idx1] = p1;
            newPoints[idx1 + 1] = interpolatedPoint;
            
            // Shift remaining points
            int offset = 0;
            for (int i = idx1 + 2; i < 6; i++)
            {
                int srcIdx = idx2 + offset;
                if (srcIdx < originalPoints.Length)
                {
                    newPoints[i] = originalPoints[srcIdx];
                }
                else
                {
                    // Not enough points for full sequence
                    return null;
                }
                offset++;
            }
            
            // Calculate deltas from new points
            for (int i = 0; i < 5; i++)
            {
                newDeltas[i] = newPoints[i + 1].Point - newPoints[i].Point;
            }
            // Target delta is from point 5 to 6 (but we only have 6 points, so use last original delta)
            if (originalDeltas.Length > idx2)
            {
                newDeltas[5] = originalDeltas[idx2];
            }
            else
            {
                return null;
            }
            
            return (newDeltas, newPoints);
        }

        private void TrainOnSequence(Vector2[] trainingDeltas, TimeSeriesPoint[] trainingPoints)
        {
            // Build input array dynamically
            List<double> inputList = new List<double>();
            
            // Add deltas (base feature)
            for(int i=0; i<5; i++)
            {
                inputList.Add(trainingDeltas[i].X / 100.0);
                inputList.Add(trainingDeltas[i].Y / 100.0);
            }
            
            // Add absolute positions if enabled
            if (_useAbsolutePosition)
            {
                for(int i=0; i<5; i++)
                {
                    inputList.Add(trainingPoints[i].Point.X / 1000.0);
                    inputList.Add(trainingPoints[i].Point.Y / 1000.0);
                }
            }
            
            // Add time deltas if enabled
            if (_useTimeDelta)
            {
                for(int i=0; i<5; i++)
                {
                    double timeDelta = trainingPoints[i+1].Time - trainingPoints[i].Time;
                    inputList.Add(timeDelta / 10.0);
                }
            }
            
            double[] nnInputs = inputList.ToArray();
            
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
            var inputPoints = points.Skip(points.Length - 5).ToArray();
            
            // Build input array dynamically
            List<double> inputList = new List<double>();
            
            // Add deltas (base feature)
            for(int i=0; i<5; i++)
            {
                inputList.Add(inputDeltas[i].X / 100.0);
                inputList.Add(inputDeltas[i].Y / 100.0);
            }
            
            // Add absolute positions if enabled
            if (_useAbsolutePosition)
            {
                for(int i=0; i<5; i++)
                {
                    inputList.Add(inputPoints[i].Point.X / 1000.0);
                    inputList.Add(inputPoints[i].Point.Y / 1000.0);
                }
            }
            
            // Add time deltas if enabled
            if (_useTimeDelta)
            {
                for(int i=0; i<4; i++)
                {
                    double timeDelta = inputPoints[i+1].Time - inputPoints[i].Time;
                    inputList.Add(timeDelta / 10.0);
                }
                // For the 5th time delta, use the average of the previous 4
                double avgTimeDelta = 0;
                for(int i=0; i<4; i++)
                {
                    avgTimeDelta += (inputPoints[i+1].Time - inputPoints[i].Time);
                }
                avgTimeDelta /= 4.0;
                inputList.Add(avgTimeDelta / 10.0);
            }
            
            double[] nnInputs = inputList.ToArray();
            
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
