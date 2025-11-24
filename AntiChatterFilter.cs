using System;
using System.Numerics;

namespace AdaptiveFilter
{
    public class AntiChatterFilter
    {
        private Vector2 _lastPos;
        private bool _first = true;

        public float Strength { get; set; } = 0.5f; // Threshold in mm (or raw units)
        public float Latency { get; set; } = 0.5f; // Multiplier for smoothing when below threshold

        public Vector2 Filter(Vector2 pos)
        {
            if (_first)
            {
                _lastPos = pos;
                _first = false;
                return pos;
            }

            float dist = Vector2.Distance(pos, _lastPos);
            
            // If movement is very small (chatter), we suppress it or smooth it heavily
            if (dist < Strength)
            {
                // Apply strong smoothing (latency)
                // Interpolate between lastPos and current pos
                // The closer to 0 dist is, the more we stick to lastPos
                
                // Simple approach:
                // If dist < Strength, we only move a fraction of the distance
                // fraction = dist / Strength * Latency
                
                float alpha = (dist / Strength) * Latency;
                if (alpha > 1.0f) alpha = 1.0f;
                
                _lastPos = Vector2.Lerp(_lastPos, pos, alpha);
            }
            else
            {
                // Movement is significant, pass through (but update lastPos)
                // To avoid snapping, we might want to blend?
                // For now, just pass through to be responsive.
                _lastPos = pos;
            }

            return _lastPos;
        }
    }
}
