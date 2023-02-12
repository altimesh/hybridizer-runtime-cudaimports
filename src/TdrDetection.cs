using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Hybridizer.Runtime.CUDAImports
{
    internal static class TdrDetection
    {
        static ITdrRegistries _tdrRegistries = TdrRegistriesFactory.GetInstance();

        static bool _IsTdrEnabledRead = false;
        static bool _TdrDelayRead = false;
        static bool _TdrEnabled = false;
        static int _TdrDelay;

        public static bool IsTdrEnabled()
        {
            if (!_IsTdrEnabledRead)
            {
                _IsTdrEnabledRead = true;
                _TdrEnabled = _tdrRegistries.TdrLevel > 0;
            }

            return _TdrEnabled;
        }

        public static int TdrDelay()
        {
            if (!_TdrDelayRead)
            {
                _TdrDelayRead = true;
                if (!IsTdrEnabled())
                {
                    _TdrDelay = -1;
                }
                else
                {
                    _TdrDelay = _tdrRegistries.TdrDelay;
                }
            }

            return _TdrDelay;
        }
    }

    internal interface ITdrRegistries
    {
        int TdrLevel { get; }
        int TdrDelay { get; }
    }

    internal static class TdrRegistriesFactory
    {
        static ITdrRegistries _instance;
        static TdrRegistriesFactory()
        {
            _instance = null;
        }

        public static ITdrRegistries GetInstance()
        {
            if (_instance == null)
            {
                if (Type.GetType("Mono.Runtime") != null)
                {
                    _instance = new LinuxTdrRegistries();
                }
                else
                {
                    _instance = new WindowsTdrRegistries();
                }
            }

            return _instance;
        }
    }

    internal class LinuxTdrRegistries : ITdrRegistries
    {
        public int TdrLevel
        {
            // TODO
            get { return 0; }
        }

        public int TdrDelay
        {
            // TODO
            get { return int.MaxValue; }
        }
    }

    internal class WindowsTdrRegistries : ITdrRegistries
    {
        bool is64BitProcess;
        bool is64BitOperatingSystem;

        public WindowsTdrRegistries()
        {
            is64BitProcess = (IntPtr.Size == 8);
            is64BitOperatingSystem = is64BitProcess || InternalCheckIsWow64();
        }

        public int TdrLevel
        {
            get
            {
                return getRegistryKey(RegistryHive.LocalMachine, "SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers", "TdrLevel");
            }
        }

        public int TdrDelay
        {
            get
            {
                return getRegistryKey(RegistryHive.LocalMachine, "SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers", "TdrDelay");
            }
        }

        private int getRegistryKey(RegistryHive hive, string path, string name) 
        {
            using (RegistryKey baseKey = RegistryKey.OpenBaseKey(hive, is64BitOperatingSystem ? RegistryView.Registry64 : RegistryView.Registry32))
            {
                using (RegistryKey subKey = baseKey.OpenSubKey(path, false))
                {
                    if (subKey != null)
                    {
                        object keyValue = subKey.GetValue(name);
                        if (keyValue is int)
                        {
                            return (int) keyValue;
                        }
                    }

                    return -1;
                }
            }
        }

        [DllImport("kernel32.dll", SetLastError = true, CallingConvention = CallingConvention.Winapi)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool IsWow64Process(
            [In] IntPtr hProcess,
            [Out] out bool wow64Process
        );

        public static bool InternalCheckIsWow64()
        {
            if ((Environment.OSVersion.Version.Major == 5 && Environment.OSVersion.Version.Minor >= 1) ||
                Environment.OSVersion.Version.Major >= 6)
            {
                using (Process p = Process.GetCurrentProcess())
                {
                    bool retVal;
                    if (!IsWow64Process(p.Handle, out retVal))
                    {
                        return false;
                    }
                    return retVal;
                }
            }
            else
            {
                return false;
            }
        }
    }
}
