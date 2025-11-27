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
                // Dynamically request interpolated sequences near the end of the buffer.
                // We'll compute a dynamic fraction based on local motion: if motion between the two
                // points is faster than average, shift the interpolation fraction toward the later point.
                double avgSpeed = 0;
                for (int i = 0; i < trainingDeltas.Length; i++) avgSpeed += trainingDeltas[i].Length();
                avgSpeed /= Math.Max(1, trainingDeltas.Length);

                // Interpolate between last two inputs (indices 4 and 5)
                {
                    double speedBetween = trainingDeltas.Length > 4 ? trainingDeltas[4].Length() : avgSpeed;
                    double fraction = 0.5 + ((speedBetween / Math.Max(1e-6, avgSpeed)) - 1.0) * 0.25;
                    fraction = Math.Clamp(fraction, 0.2, 0.8);
                    var interpolated1 = InterpolateTrainingDataAtFraction(trainingPoints, trainingDeltas, 4, 5, fraction);
                    if (interpolated1 != null)
                    {
                        TrainOnSequence(interpolated1.Value.deltas, interpolated1.Value.points);
                    }
                }

                // Interpolate between inputs 3 and 4
                {
                    double speedBetween = trainingDeltas.Length > 3 ? trainingDeltas[3].Length() : avgSpeed;
                    double fraction = 0.5 + ((speedBetween / Math.Max(1e-6, avgSpeed)) - 1.0) * 0.25;
                    fraction = Math.Clamp(fraction, 0.2, 0.8);
                    var interpolated2 = InterpolateTrainingDataAtFraction(trainingPoints, trainingDeltas, 3, 4, fraction);
                    if (interpolated2 != null)
                    {
                        TrainOnSequence(interpolated2.Value.deltas, interpolated2.Value.points);
                    }
                }
            }
        }

        private (Vector2[] deltas, TimeSeriesPoint[] points)? InterpolateTrainingData(
            TimeSeriesPoint[] originalPoints, Vector2[] originalDeltas, int idx1, int idx2)
        {
            return InterpolateTrainingDataAtFraction(originalPoints, originalDeltas, idx1, idx2, 0.5);
        }

        // More flexible interpolation: allow specifying fraction [0..1] between idx1 and idx2
        private (Vector2[] deltas, TimeSeriesPoint[] points)? InterpolateTrainingDataAtFraction(
            TimeSeriesPoint[] originalPoints, Vector2[] originalDeltas, int idx1, int idx2, double fraction)
        {
            if (idx1 < 0 || idx2 >= originalPoints.Length) return null;

            // Ensure we have a real "next" point after idx2 to avoid extrapolating
            // Training requires a real observed future delta (originalDeltas[idx2])
            // so idx2+1 must be within the bounds of originalPoints.
            if (idx2 + 1 >= originalPoints.Length) return null;

            var p1 = originalPoints[idx1];
            var p2 = originalPoints[idx2];

            // Ensure fraction is in [0,1]
            fraction = Math.Clamp(fraction, 0.0, 1.0);

            double timeDelta = p2.Time - p1.Time;
            if (timeDelta <= 0) return null;

            double targetTime = p1.Time + (timeDelta * fraction);
            float t = (float)fraction;

            Vector2 interpPoint = Vector2.Lerp(p1.Point, p2.Point, t);
            var interpolatedPoint = new TimeSeriesPoint(interpPoint, targetTime);

            // Build new sequence with interpolated point
            var newPoints = new TimeSeriesPoint[6];
            var newDeltas = new Vector2[6];

            // Copy points before interpolation
            for (int i = 0; i < idx1; i++)
            {
                newPoints[i] = originalPoints[i];
            }

            // Insert interpolated point sequence at idx1 and shift
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

            // Target delta: try to reuse original when available
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

        // Predict a short sequence of future positions without mutating the internal filters.
        // This simulates iterative predictions by using a temporary copy of the recent points
        // and appending simulated predicted points to generate realistic next-step inputs.
        public Vector2[] PredictSequence(int count, float lookahead = 1.0f)
        {
            if (!IsReady || _points.Count < 2) return Array.Empty<Vector2>();

            var tempPoints = new List<TimeSeriesPoint>(_points.ToArray());
            var results = new List<Vector2>();

            // Estimate a reasonable time step using average of recent deltas
            double avgDt = 0;
            if (tempPoints.Count >= 2)
            {
                for (int i = 0; i < tempPoints.Count - 1; i++) avgDt += (tempPoints[i + 1].Time - tempPoints[i].Time);
                avgDt /= Math.Max(1, tempPoints.Count - 1);
            }
            if (avgDt <= 0) avgDt = 1; // fallback 1 ms

            for (int k = 0; k < count; k++)
            {
                var points = tempPoints.ToArray();
                var lastPoint = points.Last();
                List<Vector2> deltas = new List<Vector2>();
                for (int i = 0; i < points.Length - 1; i++)
                {
                    deltas.Add(points[i+1].Point - points[i].Point);
                }

                if (deltas.Count < 5)
                {
                    // Not enough history to continue predicting
                    break;
                }

                var inputDeltas = deltas.Skip(deltas.Count - 5).ToArray();
                var inputPoints = points.Skip(points.Length - 5).ToArray();

                // Build input array dynamically (same as Predict, but without filters)
                List<double> inputList = new List<double>();
                for(int i=0; i<5; i++)
                {
                    inputList.Add(inputDeltas[i].X / 100.0);
                    inputList.Add(inputDeltas[i].Y / 100.0);
                }
                if (_useAbsolutePosition)
                {
                    for(int i=0; i<5; i++)
                    {
                        inputList.Add(inputPoints[i].Point.X / 1000.0);
                        inputList.Add(inputPoints[i].Point.Y / 1000.0);
                    }
                }
                if (_useTimeDelta)
                {
                    for(int i=0; i<4; i++)
                    {
                        double timeDelta = inputPoints[i+1].Time - inputPoints[i].Time;
                        inputList.Add(timeDelta / 10.0);
                    }
                    double avgTimeDelta = 0;
                    for(int i=0; i<4; i++) avgTimeDelta += (inputPoints[i+1].Time - inputPoints[i].Time);
                    avgTimeDelta /= 4.0;
                    inputList.Add(avgTimeDelta / 10.0);
                }

                double[] nnInputs = inputList.ToArray();
                var output = _nn.FeedForward(nnInputs);

                if (double.IsNaN(output[0]) || double.IsNaN(output[1]) ||
                    double.IsInfinity(output[0]) || double.IsInfinity(output[1]))
                {
                    break;
                }

                float predDeltaX = (float)(output[0] * 100.0);
                float predDeltaY = (float)(output[1] * 100.0);
                predDeltaX *= lookahead;
                predDeltaY *= lookahead;

                var nextPos = lastPoint.Point + new Vector2(predDeltaX, predDeltaY);
                results.Add(nextPos);

                // Append simulated point with estimated time so next iteration can use it
                tempPoints.Add(new TimeSeriesPoint(nextPos, lastPoint.Time + avgDt));
                // Keep the tempPoints length reasonable (simulate same capacity)
                if (tempPoints.Count > _capacity) tempPoints.RemoveAt(0);
            }

            return results.ToArray();
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
