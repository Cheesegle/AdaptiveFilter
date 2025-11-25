# AdaptiveFilter

A real-time adaptive Neural Network filter plugin for OpenTabletDriver that learns your movement patterns to provide smooth, low-latency cursor prediction.

**⚠️Note:** This project was vibe coded using AI assistance. The entire codebase, from the neural network implementation to the real-time web visualization, was developed through AI programming.

## Features

- **Neural Network Prediction**: Learns your movement patterns in real-time using a Multi-Layer Perceptron
- **Real-time Visualization**: Web-based UI showing live predictions and network structure
- **Configurable Architecture**: Adjust hidden layers, learning rate, and network complexity
- **Hybrid Mode**: Choose between pure NN prediction or hybrid real input + NN upsampling
- **Anti-Chatter Filter**: Suppress noise and reduce jitter
- **OneEuroFilter**: Additional smoothing for stable predictions
- **Training Metrics**: Monitor accuracy, output rate, and training iterations

## Installation

1. **Download or Build**:
   - Download the latest release from [Releases](../../releases)
   - Or build from source: `dotnet build -c Release`

2. **Install Plugin**:
   - Open OpenTabletDriver
   - Open Plugin Directory
   - Copy `AdaptiveFilter.dll` to this directory
   - Restart OpenTabletDriver

3. **Enable Filter**:
   - In OpenTabletDriver, go to the **Filters** tab
   - Select **AdaptiveFilter** from the list
   - Click **Apply**

## Configuration

### Basic Settings
- **Target Rate**: Output refresh rate in Hz (default: 1000)
- **Prediction Offset**: Time offset for predictions in ms (default: 0)
- **Lookahead**: Prediction multiplier (default: 1.0)

### Neural Network
- **Learning Rate**: How fast the model adapts (default: 0.01)
- **Hidden Layer Size**: Neurons per hidden layer (default: 16)
- **Hidden Layer Count**: Number of hidden layers (default: 2)
- **Samples**: Number of past points to use (default: 5)

### Filters
- **Anti-Chatter Strength**: Noise suppression threshold in mm (default: 1.0)
- **Use Hybrid Mode**: 
  - ❌ Pure NN prediction (smoothest)
  - ✅ Real input + NN upsampling (lowest latency)

### Visualization
- **Web UI Port**: Port for web interface (default: 5000)

## Web UI

Access the visualization at `http://localhost:5000`

- **Left panel**: Real-time cursor visualization (Cyan = Input, Pink = Prediction)
- **Right panel**: Live neural network structure with weighted connections
- **Stats**: Prediction offset, output rate, accuracy (mm), and training iterations

## Recommended Configurations

### For Simple/Linear Movements
- 1 layer × 12 neurons (~360 parameters)
- Lower learning rate (0.005)

### For Complex Aim Patterns (Default)
- 2 layers × 16 neurons (~2,080 parameters)
- Standard learning rate (0.01)

### For Maximum Smoothness
- 3 layers × 12 neurons (~2,000 parameters)
- Higher learning rate (0.015)

## Troubleshooting

**Web UI not loading**
- Ensure port 5000 is available
- Change `Web UI Port` in plugin settings

**Cursor stops moving**
- Reduce `Hidden Layer Count` to 1
- Lower `Hidden Layer Size` to 12

**High latency**
- Reduce `Target Rate`
- Enable `Use Hybrid Mode`

**Jittery predictions**
- Increase `Samples` count
- Increase `Anti-Chatter Strength`

## How It Works

1. **Input Processing**: Tablet reports are filtered through Anti-Chatter filter
2. **Training**: Neural network learns movement deltas from sliding window of samples
3. **Prediction**: Network predicts next position based on recent movement patterns
4. **Smoothing**: OneEuroFilter reduces jitter while maintaining responsiveness
5. **Upsampling**: Predictions are emitted at configured target rate

## Building from Source

```bash
git clone https://github.com/yourusername/AdaptiveFilter.git
cd AdaptiveFilter
dotnet build -c Release
```

Output: `bin/Release/net6.0/AdaptiveFilter.dll`

## Dependencies

- .NET 6.0
- OpenTabletDriver.Plugin
- MathNet.Numerics

## License

MIT License - See [LICENSE](LICENSE) file for details

## Acknowledgments

Built with inspiration from various prediction and smoothing techniques used in tablet driver filters.
