' Double-click: opens the drawing screen in the browser WITHOUT a black console window.
' The program stops by itself about 2 minutes after the browser tab is closed.
' If Python cannot be started, a message is shown. (The .bat file next to this one shows the error in a console window.)
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
py = sh.ExpandEnvironmentStrings("%LOCALAPPDATA%") & "\Microsoft\WindowsApps\py.exe"
If Not fso.FileExists(py) Then py = "py"
cmd = """" & py & """ -3.13 -I """ & dir & "\scripts\app.py"" --auto-exit"
rc = sh.Run(cmd, 0, True)
If rc <> 0 Then
  MsgBox "The drawing screen could not be started (Python 3.13 and 'py' are required)." & vbCrLf & "Use the .bat file next to this one to see the error message.", 48, "JW AUTO DRAW"
End If
