# 判定済みの中間データ（work\中間データ）から、全プランの図面（02_出力）だけを作り直す。
# 壁・建具の判定はやり直さない（手直しした json をそのまま使う）。直したあと全図面を検査する。
#   .\scripts\remake_all.ps1 [-Spec <仕様の名前>]
#   -Spec を付けると、その仕様で作り、検査もその仕様で行う（05_設定\現在の仕様.txt は変えない。
#   画面（app.py）が現在の仕様を切り替えても、作り直しと検査の仕様がずれないように。11/14）
param([string]$Spec = "")
$d = Split-Path $PSScriptRoot -Parent
$py = "$env:LOCALAPPDATA\Microsoft\WindowsApps\py.exe"
$env:PYTHONUTF8 = 1
# 設定は「現在の仕様」（05_設定\現在の仕様.txt）。default.json だと LIXIL の部品になる（11/13 取り違えた）
$name = if ($Spec) { $Spec } else { (Get-Content "$d\05_設定\現在の仕様.txt" -Encoding UTF8 | Select-Object -First 1).Trim() }
$env:JW_SPEC = $name  # 検査（check_conditions.py --all）も同じ仕様で
$cfg = if ($name) { "$d\05_設定\仕様\$name.json" } else { "$d\05_設定\default.json" }
"設定: $cfg"
$plans = @(
    @{ sub = "";        i1 = "p2_x31_1504x1504.png"; i2 = "p3_x51_1504x1289.png" },
    @{ sub = "プラン2"; i1 = "p2_x39_2507x1504.png"; i2 = "p3_x60_1934x1433.png" },
    @{ sub = "プラン3"; i1 = "p2_x31_4370x3725.png"; i2 = "p3_x52_1862x1791.png" },
    @{ sub = "プラン4"; i1 = "p2_x32_1648x2579.png"; i2 = "p3_x52_1504x1791.png" },
    @{ sub = "プラン5"; i1 = "p2_x31_3081x4155.png"; i2 = "p3_x52_931x2077.png" },
    @{ sub = "____________WIH"; i1 = "p2_x40_1289x2651.png"; i2 = "p3_x53_1074x1862.png" }
)
foreach ($p in $plans) {
    $s = if ($p.sub) { "\" + $p.sub } else { "" }
    $t = "$env:TEMP\jw_by_claude$s"; $mid = "$d\work\中間データ$s"; $out = "$d\02_出力$s"; $img = "$d\work\images$s"
    New-Item -ItemType Directory -Force $t | Out-Null
    Remove-Item "$t\1F.jww", "$t\2F.jww" -ErrorAction SilentlyContinue
    "== $($p.sub) 1F / 2F"
    # 階は 1F・2F（i3 があれば 3F も）。各階はすぐ上の階の壁を柱位置の判断に使う（11/15）
    $fls = @("1F", "2F"); $imgs = @($p.i1, $p.i2)
    if ($p.i3) { $fls += "3F"; $imgs += $p.i3 }
    foreach ($fl in "1F", "2F", "3F") { Remove-Item "$t\$fl.jww" -ErrorAction SilentlyContinue }
    # ユニットバスの読み取り（<階>_UB.json）がまだなければ作る（11/16 より前に判定したプラン）
    for ($k = 0; $k -lt $fls.Count; $k++) {
        if (-not (Test-Path "$mid\$($fls[$k])_UB.json")) {
            & $py -3.13 -I "$d\scripts\ub_detect.py" "$img\$($imgs[$k])" "$mid\$($fls[$k])" $cfg | Select-Object -Last 1
        }
        # 洗面台の読み取り（<階>_洗面.json）も、なければ作る（11/16）
        if (-not (Test-Path "$mid\$($fls[$k])_洗面.json")) {
            & $py -3.13 -I "$d\scripts\vanity_detect.py" "$img\$($imgs[$k])" "$mid\$($fls[$k])" $cfg | Select-Object -Last 1
        }
        # キッチンの読み取り（<階>_キッチン.json）も、なければ作る（11/16）
        if (-not (Test-Path "$mid\$($fls[$k])_キッチン.json")) {
            & $py -3.13 -I "$d\scripts\kitchen_detect.py" "$img\$($imgs[$k])" "$mid\$($fls[$k])" $cfg | Select-Object -Last 1
        }
    }
    for ($k = 0; $k -lt $fls.Count; $k++) {
        $f = $fls[$k]
        if ($k + 1 -lt $fls.Count) {
            & $py -3.13 -I "$d\scripts\make_plan.py" $cfg "$mid\${f}_walls.json" "$img\$($imgs[$k])" "$t\$f" "$mid\$($fls[$k + 1])_walls.json" > "$t\${f}_log.txt" 2>&1
        } else {
            & $py -3.13 -I "$d\scripts\make_plan.py" $cfg "$mid\${f}_walls.json" "$img\$($imgs[$k])" "$t\$f" > "$t\${f}_log.txt" 2>&1
        }
    }
    foreach ($fl in $fls) {
        if (-not (Test-Path "$t\$fl.jww")) { "  $fl を作れなかった（$t\${fl}_log.txt）"; continue }
        Copy-Item "$t\$fl.jww" "$out\$fl.jww" -Force
        Copy-Item "$t\${fl}_壁と柱.png", "$t\${fl}_建具の配置.png" "$out\" -Force
        Copy-Item "$t\${fl}_meta.json" "$mid\" -Force
    }
    # 読み取りの検査（直さず、疑いだけ）：出力\<階>_読み取りの検査.png（11/15）
    for ($k = 0; $k -lt $fls.Count; $k++) {
        & $py -3.13 -I "$d\scripts\check_reading.py" "$img\$($imgs[$k])" "$mid\$($fls[$k])" $cfg --out $out | Select-Object -Last 1
    }
}
& $py -3.13 -I "$d\scripts\check_conditions.py" --all
