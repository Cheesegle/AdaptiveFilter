using System;
using System.Runtime.InteropServices;

namespace AdaptiveFilter
{
    public static class VSyncHelper
    {
        [StructLayout(LayoutKind.Sequential)]
        private struct DWM_TIMING_INFO
        {
            public uint cbSize;
            public UNSIGNED_RATIO rateRefresh;
            public ulong qpcRefreshPeriod;
            public UNSIGNED_RATIO rateCompose;
            public ulong qpcVBlank;
            public ulong cRefresh;
            public uint cDXRefresh;
            public ulong qpcCompose;
            public ulong cFrame;
            public uint cDXPresent;
            public ulong cRefreshFrame;
            public ulong cFrameSubmitted;
            public uint cDXPresentSubmitted;
            public ulong cFrameConfirmed;
            public uint cDXPresentConfirmed;
            public ulong cRefreshConfirmed;
            public uint cDXRefreshConfirmed;
            public ulong cFramesLate;
            public uint cFramesOutstanding;
            public ulong cFrameDisplayed;
            public ulong qpcFrameDisplayed;
            public ulong cRefreshFrameDisplayed;
            public ulong cFrameComplete;
            public ulong qpcFrameComplete;
            public ulong cFramePending;
            public ulong qpcFramePending;
            public ulong cFramesDisplayed;
            public ulong cFramesComplete;
            public ulong cFramesPending;
            public ulong cFramesAvailable;
            public ulong cFramesDropped;
            public ulong cFramesMissed;
            public ulong cRefreshNextDisplayed;
            public ulong cRefreshNextPresented;
            public ulong cRefreshesDisplayed;
            public ulong cRefreshesPresented;
            public ulong cRefreshStarted;
            public ulong qpcRefreshStarted;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct UNSIGNED_RATIO
        {
            public uint uiNumerator;
            public uint uiDenominator;
        }

        [DllImport("dwmapi.dll")]
        private static extern int DwmGetCompositionTimingInfo(IntPtr hwnd, ref DWM_TIMING_INFO pTimingInfo);

        [DllImport("kernel32.dll")]
        private static extern bool QueryPerformanceCounter(out long lpPerformanceCount);

        [DllImport("kernel32.dll")]
        private static extern bool QueryPerformanceFrequency(out long lpFrequency);

        private static readonly long PerformanceFrequency;
        private static double _lastRefreshPeriodMs = 0;
        private static bool _isAvailable = true;

        static VSyncHelper()
        {
            QueryPerformanceFrequency(out PerformanceFrequency);
        }

        /// <summary>
        /// Gets the current monitor refresh rate in Hz
        /// </summary>
        public static double GetRefreshRate()
        {
            if (!_isAvailable) return 0;

            try
            {
                var timingInfo = new DWM_TIMING_INFO { cbSize = (uint)Marshal.SizeOf(typeof(DWM_TIMING_INFO)) };
                int result = DwmGetCompositionTimingInfo(IntPtr.Zero, ref timingInfo);

                if (result == 0 && timingInfo.rateRefresh.uiDenominator > 0)
                {
                    return (double)timingInfo.rateRefresh.uiNumerator / timingInfo.rateRefresh.uiDenominator;
                }
            }
            catch
            {
                _isAvailable = false;
            }

            return 0;
        }

        /// <summary>
        /// Gets the refresh period in milliseconds
        /// </summary>
        public static double GetRefreshPeriodMs()
        {
            if (!_isAvailable) return 0;

            try
            {
                var timingInfo = new DWM_TIMING_INFO { cbSize = (uint)Marshal.SizeOf(typeof(DWM_TIMING_INFO)) };
                int result = DwmGetCompositionTimingInfo(IntPtr.Zero, ref timingInfo);

                if (result == 0 && timingInfo.qpcRefreshPeriod > 0)
                {
                    _lastRefreshPeriodMs = (double)timingInfo.qpcRefreshPeriod / PerformanceFrequency * 1000.0;
                    return _lastRefreshPeriodMs;
                }
            }
            catch
            {
                _isAvailable = false;
            }

            return _lastRefreshPeriodMs;
        }

        /// <summary>
        /// Gets the time until the next VBlank in milliseconds
        /// Returns negative if VBlank has passed, positive if it's upcoming
        /// </summary>
        public static double GetTimeUntilNextVBlank()
        {
            if (!_isAvailable) return 0;

            try
            {
                var timingInfo = new DWM_TIMING_INFO { cbSize = (uint)Marshal.SizeOf(typeof(DWM_TIMING_INFO)) };
                int result = DwmGetCompositionTimingInfo(IntPtr.Zero, ref timingInfo);

                if (result == 0)
                {
                    QueryPerformanceCounter(out long currentQpc);
                    
                    double refreshPeriodMs = (double)timingInfo.qpcRefreshPeriod / PerformanceFrequency * 1000.0;
                    double lastVBlankMs = (double)timingInfo.qpcVBlank / PerformanceFrequency * 1000.0;
                    double currentMs = (double)currentQpc / PerformanceFrequency * 1000.0;
                    
                    double timeSinceLastVBlank = currentMs - lastVBlankMs;
                    double timeUntilNext = refreshPeriodMs - (timeSinceLastVBlank % refreshPeriodMs);
                    
                    return timeUntilNext;
                }
            }
            catch
            {
                _isAvailable = false;
            }

            return 0;
        }

        /// <summary>
        /// Gets timing information for optimal prediction updates
        /// Returns the optimal time offset from now to update (in ms)
        /// </summary>
        /// <param name="offsetBeforeVBlankMs">How many ms before VBlank to target (default 1.5ms)</param>
        public static double GetOptimalUpdateOffset(double offsetBeforeVBlankMs = 1.5)
        {
            double timeUntilVBlank = GetTimeUntilNextVBlank();
            
            // Update before VBlank by the specified offset to ensure the prediction is ready
            double targetOffset = timeUntilVBlank - offsetBeforeVBlankMs;
            
            // If we're very close to VBlank, wait for the next one
            if (targetOffset < 0.5)
            {
                double refreshPeriod = GetRefreshPeriodMs();
                targetOffset += refreshPeriod;
            }
            
            return Math.Max(0, targetOffset);
        }

        /// <summary>
        /// Checks if VSync information is available
        /// </summary>
        public static bool IsAvailable => _isAvailable;
    }
}
