# 判定済みの中間データ（work\中間データ）から、全プランの図面（02_出力）だけを作り直す。
# 壁・建具の判定はやり直さない（手直しした json をそのまま使う）。直したあと全図面を検査する。
#   .\scripts\remake_all.ps1
$d = Split-Path $PSScriptRoot -Parent
$py = "$env:LOCALAPPDATA\Microsoft\WindowsApps\py.exe"
$env:PYTHONUTF8 = 1
# 設定は「現在の仕様」（05_設定\現在の仕様.txt）。default.json だと LIXIL の部品になる（11/13 取り違えた）
$name = (Get-Content "$d\05_設定\現在の仕様.txt" -Encoding UTF8 | Select-Object -First 1).Trim()
$cfg = if ($name) { "$d\05_設定\仕様\$name.json" } else { "$d\05_設定\default.json" }
"設定: $cfg"
$plans = @(
    @{ sub = "";        i1 = "p2_x31_1504x1504.png"; i2 = "p3_x51_1504x1289.png" },
    @{ sub = "プラン2"; i1 = "p2_x39_2507x1504.png"; i2 = "p3_x60_1934x1433.png" },
    @{ sub = "プラン3"; i1 = "p2_x31_4370x3725.png"; i2 = "p3_x52_1862x1791.png" },
    @{ sub = "プラン4"; i1 = "p2_x32_1648x2579.png"; i2 = "p3_x52_1504x1791.png" },
    @{ sub = "プラン5"; i1 = "p2_x31_3081x4155.png"; i2 = "p3_x52_931x2077.png" }
)
foreach ($p in $plans) {
    $s = if ($p.sub) { "\" + $p.sub } else { "" }
    $t = "$env:TEMP\jw_by_claude$s"; $mid = "$d\work\中間データ$s"; $out = "$d\02_出力$s"; $img = "$d\work\images$s"
    New-Item -ItemType Directory -Force $t | Out-Null
    Remove-Item "$t\1F.jww", "$t\2F.jww" -ErrorAction SilentlyContinue
    "== $($p.sub) 1F / 2F"
    & $py -3.13 -I "$d\scripts\make_plan.py" $cfg "$mid\1F_walls.json" "$img\$($p.i1)" "$t\1F" "$mid\2F_walls.json" > "$t\1F_log.txt" 2>&1
    & $py -3.13 -I "$d\scripts\make_plan.py" $cfg "$mid\2F_walls.json" "$img\$($p.i2)" "$t\2F" > "$t\2F_log.txt" 2>&1
    foreach ($fl in "1F", "2F") {
        if (-not (Test-Path "$t\$fl.jww")) { "  $fl を作れなかった（$t\${fl}_log.txt）"; continue }
        Copy-Item "$t\$fl.jww" "$out\$fl.jww" -Force
        Copy-Item "$t\${fl}_壁と柱.png", "$t\${fl}_建具の配置.png" "$out\" -Force
        Copy-Item "$t\${fl}_meta.json" "$mid\" -Force
    }
}
& $py -3.13 -I "$d\scripts\check_conditions.py" --all
