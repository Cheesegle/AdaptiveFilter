using System;
using System.Runtime.InteropServices;
using System.Numerics;

namespace AdaptiveFilter
{
    public static class InputInjector
    {
        [DllImport("user32.dll")]
        private static extern uint SendInput(uint nInputs, [MarshalAs(UnmanagedType.LPArray), In] INPUT[] pInputs, int cbSize);

        [DllImport("user32.dll")]
        private static extern int GetSystemMetrics(int nIndex);

        private const int SM_XVIRTUALSCREEN = 76;
        private const int SM_YVIRTUALSCREEN = 77;
        private const int SM_CXVIRTUALSCREEN = 78;
        private const int SM_CYVIRTUALSCREEN = 79;

        private const uint INPUT_MOUSE = 0;
        private const uint MOUSEEVENTF_MOVE = 0x0001;
        private const uint MOUSEEVENTF_ABSOLUTE = 0x8000;
        private const uint MOUSEEVENTF_VIRTUALDESK = 0x4000;

        [StructLayout(LayoutKind.Sequential)]
        private struct INPUT
        {
            public uint type;
            public MOUSEINPUT mi;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct MOUSEINPUT
        {
            public int dx;
            public int dy;
            public uint mouseData;
            public uint dwFlags;
            public uint time;
            public IntPtr dwExtraInfo;
        }

        private static double _screenWidth;
        private static double _screenHeight;
        private static double _screenX;
        private static double _screenY;
        private static bool _initialized;

        private static void Initialize()
        {
            _screenX = GetSystemMetrics(SM_XVIRTUALSCREEN);
            _screenY = GetSystemMetrics(SM_YVIRTUALSCREEN);
            _screenWidth = GetSystemMetrics(SM_CXVIRTUALSCREEN);
            _screenHeight = GetSystemMetrics(SM_CYVIRTUALSCREEN);
            _initialized = true;
        }

        public static void MoveMouse(Vector2 position)
        {
            if (!_initialized) Initialize();

            // Convert absolute pixel coordinates to normalized absolute coordinates (0..65535)
            // relative to the virtual desktop.
            
            // Formula: (Coordinate - Origin) / Size * 65535
            double normalizedX = (position.X - _screenX) / _screenWidth * 65535.0;
            double normalizedY = (position.Y - _screenY) / _screenHeight * 65535.0;

            INPUT[] inputs = new INPUT[1];
            inputs[0].type = INPUT_MOUSE;
            inputs[0].mi.dx = (int)normalizedX;
            inputs[0].mi.dy = (int)normalizedY;
            inputs[0].mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK;

            SendInput(1, inputs, Marshal.SizeOf(typeof(INPUT)));
        }
    }
}
