// LiveTranslate.exe: the double-click entry, without any console window.
//   Installed (.venv\Scripts\pythonw.exe, .venv\.livetranslate-ready and user_settings.json)
//     -> starts ".venv\Scripts\pythonw.exe main.py" and exits.
//   Otherwise -> the installation window: runs launcher\setup.ps1 -Yes hidden and shows its output;
//     when it succeeds, LiveTranslate starts.
// Options (tests / troubleshooting): --probe   exit code 0 = installed, 2 = not installed (no window)
//                                    --setup   show the installation window even when installed
//                                    --install start installing at once (no "start" click)
// Built by launcher\build_exe.ps1 with Windows' own C# compiler (.NET Framework 4, C# 5).
using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms;

[assembly: AssemblyTitle("LiveTranslate")]
[assembly: AssemblyProduct("LiveTranslate")]
[assembly: AssemblyDescription("LiveTranslate launcher and installer")]

namespace LiveTranslateLauncher
{
    static class Program
    {
        [DllImport("user32.dll")]
        static extern bool SetProcessDPIAware();

        [STAThread]
        static int Main(string[] args)
        {
            string root = AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\');
            bool installed = IsInstalled(root);
            if (Array.IndexOf(args, "--probe") >= 0)
                return installed ? 0 : 2;
            if (installed && Array.IndexOf(args, "--setup") < 0 && Array.IndexOf(args, "--install") < 0)
                return StartApp(root) ? 0 : 1;

            SetProcessDPIAware();
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            SetupForm form = new SetupForm(root, Array.IndexOf(args, "--install") >= 0);
            Application.Run(form);
            return form.Succeeded ? 0 : 1;
        }

        public static string PythonW(string root)
        {
            return Path.Combine(root, @".venv\Scripts\pythonw.exe");
        }

        public static bool IsInstalled(string root)
        {
            return File.Exists(PythonW(root))
                && File.Exists(Path.Combine(root, @".venv\.livetranslate-ready"))
                && File.Exists(Path.Combine(root, "user_settings.json"));
        }

        public static bool StartApp(string root)
        {
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo(PythonW(root), "main.py");
                psi.WorkingDirectory = root;
                psi.UseShellExecute = false;
                using (Process p = Process.Start(psi))
                {
                    // pythonw shows nothing when it fails at once (a broken package, a syntax error):
                    // wait a moment and say where to look instead of silently doing nothing
                    if (p.WaitForExit(10000) && p.HasExited && p.ExitCode != 0)
                    {
                        MessageBox.Show("LiveTranslate 啟動後馬上結束了（結束代碼 " + p.ExitCode + "）。\r\n"
                                        + "請改用「LiveTranslate（除錯）.bat」啟動，黑色視窗會顯示原因；"
                                        + "記錄也在 logs 資料夾。", "LiveTranslate",
                                        MessageBoxButtons.OK, MessageBoxIcon.Error);
                        return false;
                    }
                }
                return true;
            }
            catch (Exception e)
            {
                MessageBox.Show("無法啟動 LiveTranslate：" + e.Message, "LiveTranslate",
                                MessageBoxButtons.OK, MessageBoxIcon.Error);
                return false;
            }
        }
    }

    class SetupForm : Form
    {
        readonly string root;
        readonly Label intro;
        readonly Label step;
        readonly TextBox output;
        readonly ProgressBar bar;
        readonly Button startButton;
        readonly Button logButton;
        readonly Button closeButton;
        Process proc;
        bool finished;  // after the exit, late output lines must not replace the final message
        public bool Succeeded;

        public SetupForm(string root, bool startNow)
        {
            this.root = root;
            Text = "安裝 LiveTranslate";
            Font = new Font("Microsoft JhengHei UI", 9.75f);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(620, 500);
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            StartPosition = FormStartPosition.CenterScreen;
            Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath);  // the exe's own icon

            intro = new Label();
            intro.Location = new Point(18, 14);
            intro.Size = new Size(584, 140);  // a long install path wraps
            intro.Text = "第一次使用需要安裝：Python、Git、Ollama（缺的才裝）、程式套件（約 5 GB）與語音辨識模型。"
                       + "依網路速度約 20–40 分鐘，最多下載約 12 GB。\r\n"
                       + "安裝位置：" + root + "\r\n"
                       + "過程中不用操作；中途失敗的話，再開一次 LiveTranslate 會從沒完成的地方繼續。";
            Controls.Add(intro);

            step = new Label();
            step.Location = new Point(18, 158);
            step.Size = new Size(584, 44);  // two lines for a failure message
            step.Font = new Font(Font, FontStyle.Bold);
            Controls.Add(step);

            bar = new ProgressBar();
            bar.Location = new Point(18, 206);
            bar.Size = new Size(584, 12);
            bar.Style = ProgressBarStyle.Marquee;
            bar.MarqueeAnimationSpeed = 0;
            Controls.Add(bar);

            output = new TextBox();
            output.Location = new Point(18, 226);
            output.Size = new Size(584, 214);
            output.Multiline = true;
            output.ReadOnly = true;
            output.ScrollBars = ScrollBars.Vertical;
            output.Font = new Font("Microsoft JhengHei UI", 8.5f);
            Controls.Add(output);

            logButton = new Button();
            logButton.Text = "開啟記錄資料夾";
            logButton.Location = new Point(18, 452);
            logButton.Size = new Size(150, 34);
            logButton.Visible = false;
            logButton.Click += delegate { OpenLogs(); };
            Controls.Add(logButton);

            startButton = new Button();
            startButton.Text = "開始安裝";
            startButton.Location = new Point(366, 452);
            startButton.Size = new Size(130, 34);
            startButton.Click += delegate { StartSetup(); };
            Controls.Add(startButton);
            AcceptButton = startButton;

            closeButton = new Button();
            closeButton.Text = "取消";
            closeButton.Location = new Point(506, 452);
            closeButton.Size = new Size(96, 34);
            closeButton.Click += delegate { Close(); };
            Controls.Add(closeButton);

            if (startNow)
                Shown += delegate { StartSetup(); };
        }

        void StartSetup()
        {
            string script = Path.Combine(root, @"launcher\setup.ps1");
            if (!File.Exists(script))
            {
                Fail("找不到安裝程式：" + script);
                return;
            }
            finished = false;
            startButton.Enabled = false;
            logButton.Visible = false;
            closeButton.Text = "中止";
            step.ForeColor = SystemColors.ControlText;
            step.Text = "準備安裝…";
            bar.Style = ProgressBarStyle.Marquee;
            bar.MarqueeAnimationSpeed = 30;
            AddLine("> 開始安裝（" + DateTime.Now.ToString("HH:mm:ss") + "）");

            // UTF-8 for PowerShell's own output and for the Python / pip it runs (read as UTF-8 here)
            string command = "[Console]::OutputEncoding = [Text.Encoding]::UTF8; & '"
                           + script.Replace("'", "''") + "' -Yes; exit $LASTEXITCODE";
            ProcessStartInfo psi = new ProcessStartInfo("powershell.exe",
                "-NoProfile -ExecutionPolicy Bypass -Command \"" + command + "\"");
            psi.WorkingDirectory = root;
            psi.UseShellExecute = false;
            psi.CreateNoWindow = true;
            psi.RedirectStandardOutput = true;
            psi.RedirectStandardError = true;
            psi.StandardOutputEncoding = Encoding.UTF8;
            psi.StandardErrorEncoding = Encoding.UTF8;
            psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
            proc = new Process();
            proc.StartInfo = psi;
            proc.EnableRaisingEvents = true;
            proc.OutputDataReceived += OnData;
            proc.ErrorDataReceived += OnData;
            proc.Exited += delegate { BeginInvoke(new Action(Finished)); };
            try
            {
                proc.Start();
            }
            catch (Exception e)
            {
                proc = null;
                Fail("無法執行安裝程式：" + e.Message);
                return;
            }
            proc.BeginOutputReadLine();
            proc.BeginErrorReadLine();
        }

        void OnData(object sender, DataReceivedEventArgs e)
        {
            if (e.Data != null)
                BeginInvoke(new Action<string>(AddLine), e.Data);
        }

        void AddLine(string line)
        {
            output.AppendText(line + "\r\n");
            // setup.ps1 starts each step with "[HH:mm:ss] what it does"
            string t = line.Trim();
            if (!finished && t.Length > 11 && t[0] == '[' && t[3] == ':' && t[6] == ':' && t[9] == ']')
                step.Text = t.Substring(10).Trim();
        }

        void Finished()
        {
            if (proc == null)
                return;
            finished = true;
            proc.WaitForExit();  // the last output lines arrive before this returns
            int code = proc.ExitCode;
            proc.Dispose();
            proc = null;
            if (code == 0 && Program.IsInstalled(root))
            {
                Succeeded = true;
                bar.Style = ProgressBarStyle.Blocks;
                bar.Value = bar.Maximum;
                step.Text = "安裝完成，正在啟動 LiveTranslate…";
                closeButton.Enabled = false;
                if (Program.StartApp(root))
                {
                    Timer t = new Timer();
                    t.Interval = 1500;
                    t.Tick += delegate { t.Stop(); Close(); };
                    t.Start();
                }
                else
                {
                    closeButton.Enabled = true;
                    closeButton.Text = "關閉";
                }
                return;
            }
            Fail(code == 0 ? "安裝程式結束了，但安裝還不完整。請按「重試」。"
                           : "安裝失敗（結束代碼 " + code + "）。原因在上方的訊息與記錄檔；修正後按「重試」，會從沒完成的地方繼續。");
        }

        void Fail(string message)
        {
            bar.MarqueeAnimationSpeed = 0;
            bar.Style = ProgressBarStyle.Blocks;
            bar.Value = 0;
            step.ForeColor = Color.Firebrick;
            step.Text = message;
            AddLine("[失敗] " + message);
            startButton.Text = "重試";
            startButton.Enabled = true;
            logButton.Visible = true;
            closeButton.Text = "關閉";
            closeButton.Enabled = true;
        }

        void OpenLogs()
        {
            string logs = Path.Combine(root, "logs");
            try
            {
                Directory.CreateDirectory(logs);
                Process.Start("explorer.exe", "\"" + logs + "\"");
            }
            catch (Exception e)
            {
                MessageBox.Show("無法開啟 " + logs + "：" + e.Message, Text, MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            if (proc != null && !proc.HasExited)
            {
                DialogResult r = MessageBox.Show("安裝還在進行，要中止嗎？下次開啟 LiveTranslate 會從沒完成的地方繼續。",
                                                 Text, MessageBoxButtons.YesNo, MessageBoxIcon.Question,
                                                 MessageBoxDefaultButton.Button2);
                if (r != DialogResult.Yes)
                {
                    e.Cancel = true;
                    return;
                }
                KillTree(proc.Id);
            }
            base.OnFormClosing(e);
        }

        void KillTree(int pid)
        {
            // setup.ps1 runs pip / winget below it: end the whole tree, like closing a console did
            try
            {
                ProcessStartInfo k = new ProcessStartInfo("taskkill.exe", "/T /F /PID " + pid);
                k.UseShellExecute = false;
                k.CreateNoWindow = true;
                using (Process p = Process.Start(k))
                    p.WaitForExit(10000);
            }
            catch (Exception ex)
            {
                MessageBox.Show("無法中止安裝程式：" + ex.Message, Text, MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }
    }
}
