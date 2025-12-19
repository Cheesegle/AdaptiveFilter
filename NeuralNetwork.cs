using System;
using System.Linq;

namespace AdaptiveFilter
{
    public class NeuralNetwork
    {
        private readonly int[] _layers;
        private readonly double[][] _neurons;
        private readonly double[][] _biases;
        private readonly double[][][] _weights;
        private readonly Random _random = new Random();

        public int[] Layers => _layers;

        public NeuralNetwork(params int[] layers)
        {
            _layers = layers;
            _neurons = new double[layers.Length][];
            _biases = new double[layers.Length][];
            _weights = new double[layers.Length - 1][][];

            for (int i = 0; i < layers.Length; i++)
            {
                _neurons[i] = new double[layers[i]];
                _biases[i] = new double[layers[i]];
                
                if (i > 0)
                {
                    _weights[i - 1] = new double[layers[i]][];
                    for (int j = 0; j < layers[i]; j++)
                    {
                        _weights[i - 1][j] = new double[layers[i - 1]];
                    }
                }
            }

            InitializeWeights();
        }

        private void InitializeWeights()
        {
            for (int i = 0; i < _weights.Length; i++)
            {
                double range = Math.Sqrt(2.0 / (_layers[i] + _layers[i+1])); // Xavier/He-ish initialization
                for (int j = 0; j < _weights[i].Length; j++)
                {
                    _biases[i + 1][j] = (_random.NextDouble() * 2 - 1) * 0.01; // Small biases
                    for (int k = 0; k < _weights[i][j].Length; k++)
                    {
                        _weights[i][j][k] = (_random.NextDouble() * 2 - 1) * range;
                    }
                }
            }
        }

        public double[] FeedForward(double[] inputs)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                _neurons[0][i] = inputs[i];
            }

            for (int i = 1; i < _layers.Length; i++)
            {
                bool isOutputLayer = (i == _layers.Length - 1);
                for (int j = 0; j < _neurons[i].Length; j++)
                {
                    double value = 0;
                    for (int k = 0; k < _neurons[i - 1].Length; k++)
                    {
                        value += _weights[i - 1][j][k] * _neurons[i - 1][k];
                    }
                    
                    double z = value + _biases[i][j];
                    _neurons[i][j] = isOutputLayer ? z : Activate(z);
                }
            }

            return _neurons[_layers.Length - 1];
        }

        public void BackPropagate(double[] inputs, double[] expected, double learningRate)
        {
            FeedForward(inputs);

            double[][] gammas = new double[_layers.Length][];
            for (int i = 0; i < _layers.Length; i++)
            {
                gammas[i] = new double[_layers[i]];
            }

            int layer = _layers.Length - 1;
            for (int i = 0; i < _neurons[layer].Length; i++)
            {
                // For linear output, ActivateDer = 1.0
                gammas[layer][i] = (_neurons[layer][i] - expected[i]);
            }

            for (int i = _layers.Length - 2; i > 0; i--)
            {
                layer = i;
                for (int j = 0; j < _neurons[layer].Length; j++)
                {
                    double gamma = 0;
                    for (int k = 0; k < gammas[layer + 1].Length; k++)
                    {
                        gamma += gammas[layer + 1][k] * _weights[layer][k][j];
                    }
                    gammas[layer][j] = gamma * ActivateDer(_neurons[layer][j]);
                }
            }

            for (int i = 0; i < _weights.Length; i++)
            {
                for (int j = 0; j < _weights[i].Length; j++)
                {
                    // Gradient clipping
                    double biasDelta = gammas[i + 1][j] * learningRate;
                    _biases[i + 1][j] -= Math.Clamp(biasDelta, -0.1, 0.1);
                    
                    for (int k = 0; k < _weights[i][j].Length; k++)
                    {
                        double weightDelta = gammas[i + 1][j] * _neurons[i][k] * learningRate;
                        _weights[i][j][k] -= Math.Clamp(weightDelta, -0.1, 0.1);
                    }
                }
            }
        }

        private double Activate(double value)
        {
            return Math.Tanh(value); // Tanh for -1 to 1 range
        }

        private double ActivateDer(double value)
        {
            return 1 - (value * value); // Derivative of Tanh
        }
        public double[] GetWeights()
        {
            // Calculate total weights
            int count = 0;
            for (int i = 0; i < _weights.Length; i++)
            {
                for (int j = 0; j < _weights[i].Length; j++)
                {
                    count += _weights[i][j].Length;
                }
            }

            double[] flatWeights = new double[count];
            int index = 0;
            for (int i = 0; i < _weights.Length; i++)
            {
                for (int j = 0; j < _weights[i].Length; j++)
                {
                    for (int k = 0; k < _weights[i][j].Length; k++)
                    {
                        flatWeights[index++] = _weights[i][j][k];
                    }
                }
            }
            return flatWeights;
        }
    }
}
