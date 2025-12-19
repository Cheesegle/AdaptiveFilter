using System;
using System.Numerics;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;

namespace AdaptiveFilter
{
    public class PredictionCore
    {
        private readonly int _capacity;
        private readonly Queue<TimeSeriesPoint> _points;
        private TimeSeriesPoint? _lastPredictedOutput = null;
        private NeuralNetwork _nn = null!;
        private readonly OneEuroFilter _filterX;
        private readonly OneEuroFilter _filterY;
        
        // Thread-safe collection for reinforcement points visualization
        private readonly ConcurrentQueue<Vector2> _reinforcementPoints = new();
        private const int MaxReinforcementPoints = 150;

        public bool IsReady => _points.Count >= _capacity;
        public int Complexity { get; set; } = 2; 
        public int[] LayerSizes => _nn.Layers;
        public int TrainingIterations { get; private set; } = 0; 

        public double LearningRate { get; set; } = 0.1;
        
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

        private bool _usePredictedInput = true;
        public bool UsePredictedInput
        {
            get => _usePredictedInput;
            set => _usePredictedInput = value;
        }

        public PredictionCore(int capacity = 10)
        {
            _capacity = capacity;
            _points = new Queue<TimeSeriesPoint>(capacity);
            
            RebuildNetwork();
            
            _filterX = new OneEuroFilter(2.0, 0.001);
            _filterY = new OneEuroFilter(2.0, 0.001);
        }

        private void RebuildNetwork()
        {
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
            TrainingIterations = 0; 
        }

        public void Add(Vector2 point, double time)
        {
            if (_points.Count >= _capacity)
            {
                _points.Dequeue();
            }
            _points.Enqueue(new TimeSeriesPoint(point, time));
            
            // Clear last predicted output since we have a new raw input
            _lastPredictedOutput = null;

            if (IsReady)
            {
                Train();
            }
        }
        
        public void AddPredictedOutput(Vector2 point, double time)
        {
            _lastPredictedOutput = new TimeSeriesPoint(point, time);
        }

        public Vector2[] GetReinforcementPoints()
        {
            lock (_reinforcementPoints)
            {
                return _reinforcementPoints.ToArray();
            }
        }

        private void Train()
        {
            var points = _points.ToArray();
            int numDeltas = points.Length - 1;
            if (numDeltas < 6) return;

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
            
            // Interpolated training - ONLY between P3→P4 (3rd and 2nd most recent)
            if (_useInterpolatedTraining && trainingPoints.Length >= 6)
            {
                // Check for "bad" interpolation segment (jitter/noise)
                Vector2 p3 = trainingPoints[3].Point;
                Vector2 p4 = trainingPoints[4].Point;
                Vector2 p2 = trainingPoints[2].Point;
                
                Vector2 segmentDelta = p4 - p3;
                Vector2 historyDelta = p3 - p2; 
                
                bool isWrongDirection = Vector2.Dot(segmentDelta, historyDelta) < 0;
                bool isTooClose = segmentDelta.Length() < 0.5f;
                
                Vector2? punishmentTarget = null;
                float learningRateScale = 1.0f;
                
                if (isWrongDirection && isTooClose)
                {
                    punishmentTarget = Vector2.Zero;
                    learningRateScale = 0.5f;
                }
                
                double[] fractions = { 0.33, 0.5, 0.67 };
                
                foreach (double fraction in fractions)
                {
                    var interpolated = InterpolateTrainingDataAtFraction(trainingPoints, trainingDeltas, 3, 4, fraction);
                    if (interpolated != null)
                    {
                        TrainOnSequence(interpolated.Value.deltas, interpolated.Value.points, punishmentTarget, learningRateScale);
                    }
                }
            }
        }

        private (Vector2[] deltas, TimeSeriesPoint[] points)? InterpolateTrainingDataAtFraction(
            TimeSeriesPoint[] originalPoints, Vector2[] originalDeltas, int idx1, int idx2, double fraction)
        {
            if (idx1 < 0 || idx2 >= originalPoints.Length) return null;
            if (idx2 + 1 >= originalPoints.Length) return null;

            var p1 = originalPoints[idx1];
            var p2 = originalPoints[idx2];

            fraction = Math.Clamp(fraction, 0.0, 1.0);

            double timeDelta = p2.Time - p1.Time;
            if (timeDelta <= 0) return null;

            double targetTime = p1.Time + (timeDelta * fraction);
            float t = (float)fraction;

            Vector2 interpPoint = Vector2.Lerp(p1.Point, p2.Point, t);
            var interpolatedPoint = new TimeSeriesPoint(interpPoint, targetTime);

            var newPoints = new TimeSeriesPoint[6];
            var newDeltas = new Vector2[6];

            for (int i = 0; i < idx1; i++)
            {
                newPoints[i] = originalPoints[i];
            }

            newPoints[idx1] = p1;
            newPoints[idx1 + 1] = interpolatedPoint;

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
                    return null;
                }
                offset++;
            }

            for (int i = 0; i < 5; i++)
            {
                newDeltas[i] = newPoints[i + 1].Point - newPoints[i].Point;
            }

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

        private void TrainOnSequence(Vector2[] trainingDeltas, TimeSeriesPoint[] trainingPoints, Vector2? targetOverride = null, float learningRateScale = 1.0f)
        {
            List<double> inputList = new List<double>();
            
            for(int i=0; i<5; i++)
            {
                inputList.Add(trainingDeltas[i].X / 10.0);
                inputList.Add(trainingDeltas[i].Y / 10.0);
            }
            
            if (_useAbsolutePosition)
            {
                for(int i=0; i<5; i++)
                {
                    inputList.Add(trainingPoints[i].Point.X / 1000.0);
                    inputList.Add(trainingPoints[i].Point.Y / 1000.0);
                }
            }
            
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
            if (targetOverride.HasValue)
            {
                targets[0] = targetOverride.Value.X / 10.0;
                targets[1] = targetOverride.Value.Y / 10.0;
            }
            else
            {
                targets[0] = trainingDeltas[5].X / 10.0;
                targets[1] = trainingDeltas[5].Y / 10.0;
            }
            
            _nn.BackPropagate(nnInputs, targets, LearningRate * learningRateScale);
            TrainingIterations++;
            
            if (trainingPoints.Length >= 6)
            {
                Vector2 reinforcementPos = trainingPoints[5].Point;
                lock (_reinforcementPoints)
                {
                    _reinforcementPoints.Enqueue(reinforcementPos);
                    while (_reinforcementPoints.Count > MaxReinforcementPoints)
                    {
                        _reinforcementPoints.TryDequeue(out _);
                    }
                    if (_reinforcementPoints.Count % 10 == 0)
                    {
                        Console.WriteLine($"Reinforcement points: {_reinforcementPoints.Count}");
                    }
                }
            }
        }

        public Vector2 Predict(double targetTime, int steps = 1, float gain = 1.0f)
        {
            if (!IsReady || _points.Count < 2) return _points.LastOrDefault().Point;

            // Start with current points
            var tempPoints = new List<TimeSeriesPoint>(_points);
            if (_usePredictedInput && _lastPredictedOutput.HasValue)
            {
                tempPoints.Add(_lastPredictedOutput.Value);
            }
            
            // Calculate avgDt for time delta projection
            double avgDt = 0;
            if (tempPoints.Count >= 2)
            {
                for (int i = 0; i < tempPoints.Count - 1; i++) avgDt += (tempPoints[i + 1].Time - tempPoints[i].Time);
                avgDt /= Math.Max(1, tempPoints.Count - 1);
            }
            if (avgDt <= 0) avgDt = 1;

            Vector2 lastRetPos = tempPoints.Last().Point;

            for (int k = 0; k < steps; k++)
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
                    // If we have at least 1 delta, we can try to use it if the NN could handle it, 
                    // but our NN is fixed at 5 deltas. So we must break.
                    lastRetPos = lastPoint.Point;
                    break;
                }
                
                var inputDeltas = deltas.Skip(deltas.Count - 5).ToArray();
                var inputPoints = points.Skip(points.Length - 5).ToArray();
                
                List<double> inputList = new List<double>();
                for(int i=0; i<5; i++)
                {
                    inputList.Add(inputDeltas[i].X / 10.0);
                    inputList.Add(inputDeltas[i].Y / 10.0);
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
                    double currentAvgDt = 0;
                    for(int i=0; i<4; i++) currentAvgDt += (inputPoints[i+1].Time - inputPoints[i].Time);
                    currentAvgDt /= 4.0;
                    inputList.Add(currentAvgDt / 10.0);
                }
                
                double[] nnInputs = inputList.ToArray();
                var output = _nn.FeedForward(nnInputs);
                
                if (double.IsNaN(output[0]) || double.IsNaN(output[1]) || 
                    double.IsInfinity(output[0]) || double.IsInfinity(output[1]))
                {
                    lastRetPos = lastPoint.Point;
                    break;
                }
                
                float predDeltaX = (float)(output[0] * 10.0) * gain;
                float predDeltaY = (float)(output[1] * 10.0) * gain;
                
                lastRetPos = lastPoint.Point + new Vector2(predDeltaX, predDeltaY);
                
                // For the next step, push this predicted point into history
                tempPoints.Add(new TimeSeriesPoint(lastRetPos, lastPoint.Time + avgDt));
                // Keep only the last required points for the sliding window
                int requiredPoints = 6; // 5 deltas = 6 points
                if (_usePredictedInput) requiredPoints++;
                if (tempPoints.Count > requiredPoints) tempPoints.RemoveAt(0);
            }

            return lastRetPos;
        }

        public Vector2[] PredictSequence(int count, int stepsPerPoint = 1, float gain = 1.0f)
        {
            if (!IsReady || _points.Count < 2) return Array.Empty<Vector2>();

            // Start with current points
            var tempPoints = new List<TimeSeriesPoint>(_points);
            if (_usePredictedInput && _lastPredictedOutput.HasValue)
            {
                tempPoints.Add(_lastPredictedOutput.Value);
            }
            var results = new List<Vector2>();

            double avgDt = 0;
            if (tempPoints.Count >= 2)
            {
                for (int i = 0; i < tempPoints.Count - 1; i++) avgDt += (tempPoints[i + 1].Time - tempPoints[i].Time);
                avgDt /= Math.Max(1, tempPoints.Count - 1);
            }
            if (avgDt <= 0) avgDt = 1; 

            for (int k = 0; k < count; k++)
            {
                Vector2 currentPos = Vector2.Zero;
                
                // Step ahead multiple times for each visual point if requested
                for (int s = 0; s < stepsPerPoint; s++)
                {
                    var points = tempPoints.ToArray();
                    var lastPoint = points.Last();
                    List<Vector2> deltas = new List<Vector2>();
                    for (int i = 0; i < points.Length - 1; i++)
                    {
                        deltas.Add(points[i+1].Point - points[i].Point);
                    }

                    if (deltas.Count < 5) break;

                    var inputDeltas = deltas.Skip(deltas.Count - 5).ToArray();
                    var inputPoints = points.Skip(points.Length - 5).ToArray();

                    List<double> inputList = new List<double>();
                    for(int i=0; i<5; i++)
                    {
                        inputList.Add(inputDeltas[i].X / 10.0);
                        inputList.Add(inputDeltas[i].Y / 10.0);
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
                        double currentAvgDt = 0;
                        for(int i=0; i<4; i++) currentAvgDt += (inputPoints[i+1].Time - inputPoints[i].Time);
                        currentAvgDt /= 4.0;
                        inputList.Add(currentAvgDt / 10.0);
                    }

                    double[] nnInputs = inputList.ToArray();
                    var output = _nn.FeedForward(nnInputs);

                    if (double.IsNaN(output[0]) || double.IsNaN(output[1]) ||
                        double.IsInfinity(output[0]) || double.IsInfinity(output[1]))
                    {
                        break;
                    }

                    float predDeltaX = (float)(output[0] * 10.0) * gain;
                    float predDeltaY = (float)(output[1] * 10.0) * gain;

                    currentPos = lastPoint.Point + new Vector2(predDeltaX, predDeltaY);
                    
                    tempPoints.Add(new TimeSeriesPoint(currentPos, lastPoint.Time + avgDt));
                    int requiredPoints = 6;
                    if (_usePredictedInput) requiredPoints++;
                    if (tempPoints.Count > requiredPoints) tempPoints.RemoveAt(0);
                }
                
                if (currentPos == Vector2.Zero) break;
                results.Add(currentPos);
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
