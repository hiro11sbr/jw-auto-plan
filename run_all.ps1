# プランPDFから平面詳細図（.jww）と確認画像をすべて作り直す。
#   入力 : 01_入力\plan_20261012.pdf
#   出力 : 02_出力\  （1F.jww / 2F.jww と確認画像）
#   中間 : work\中間データ\  （判定結果のjsonや文字を消した画像など。見なくてOK）
# 処理の順番： 文字の検出 → 文字の削除 → 壁芯・壁の判定 → 建具の判定 → 壁のオフセット・柱・jww
# ezjww の保存はこのフォルダ直下だと失敗するため、いったん %TEMP% に書いてからコピーする。
# 別のプランを通すとき（例）:
#   .\run_all.ps1 -Pdf 01_入力\プラン2.pdf -Name プラン2 -Img1F p2_x39_2507x1504.png -Img2F p3_x60_1934x1433.png
#   出力は 02_出力\プラン2\、中間は work\中間データ\プラン2\。図面の画像名は extract_plans.py の表示で確かめる
param(
    [string]$Pdf = "01_入力\plan_20261012.pdf",
    [string]$Name = "",
    [string]$Img1F = "p2_x31_1504x1504.png",
    [string]$Img2F = "p3_x51_1504x1289.png",
    [string]$Img3F = ""   # 3階建ての時だけ、3階の図の画像名（例 p4_x50_1504x1289.png）。1F・2F・3F の順に処理する（11/15）
)
$d = $PSScriptRoot
$py = "$env:LOCALAPPDATA\Microsoft\WindowsApps\py.exe"
$sub = if ($Name) { "\$Name" } else { "" }
$t = "$env:TEMP\jw_by_claude$sub"
$out = "$d\02_出力$sub"
$mid = "$d\work\中間データ$sub"
$img_dir = "$d\work\images$sub"
New-Item -ItemType Directory -Force $t, $out, $mid, $img_dir | Out-Null
# 判定（文字・壁・建具）は default.json。図面を書く make_plan だけは「現在の仕様」（05_設定\現在の仕様.txt。
# なければ default.json）。make_plan に default.json を渡すと LIXIL の部品になる（11/13）
$cfg = "$d\05_設定\default.json"
$specName = if (Test-Path "$d\05_設定\現在の仕様.txt") { (Get-Content "$d\05_設定\現在の仕様.txt" -Encoding UTF8 | Select-Object -First 1).Trim() } else { "" }
$planCfg = if ($specName -and (Test-Path "$d\05_設定\仕様\$specName.json")) { "$d\05_設定\仕様\$specName.json" } else { $cfg }

# 寸法の文字の読み取り（RapidOCR）が入っていなければ、初回だけ自動で入れる（2026-11-03）
& $py -3.13 -I -c "import rapidocr_onnxruntime" 2>$null
if ($LASTEXITCODE -ne 0) {
    "RapidOCR を入れます（初回だけ、数分かかることがあります）"
    # onnxruntime 1.32 はこのPC（Windows 10）で読み込みが止まったので、動いた 1.20.1 に固定（11/13）
    & $py -3.13 -m pip install rapidocr_onnxruntime "onnxruntime==1.20.1"
}

& $py -3.13 -I "$d\scripts\extract_plans.py" "$d\$Pdf" $img_dir
$img = @{ "1F" = "$img_dir\$Img1F"; "2F" = "$img_dir\$Img2F" }
$floors = @("1F", "2F")
if ($Img3F) { $img["3F"] = "$img_dir\$Img3F"; $floors += "3F" }

$names = "$d\05_設定\部屋名リスト.csv"
foreach ($fl in $floors) {
    # 0. 部屋名の読み取り（文字を消す前に。Windows 標準の日本語OCR＋部屋名リストで照合）
    & $py -3.13 -I "$d\scripts\read_room_names.py" $img[$fl] $cfg $names "$mid\$fl" | Select-Object -First 1
    # 1. 文字の検出と削除（赤字・黒字を消して、きれいな画像を作る）
    & $py -3.13 -I "$d\scripts\clean_text.py" $img[$fl] $cfg "$mid\$fl"
    $clean = "$mid\${fl}_文字削除後.png"
    # 2. 壁芯・壁の判定（455・227.5 の線ごと）。元の画像で判定する
    #    （文字を消した画像だと、文字が重なっていた壁まで欠けて精度が下がった 2026-10-22）
    #    文字を消した画像は「文字のある所」を知るためだけに渡す（文字を建具の線と見誤らない）
    # 1b. 寸法線（寸法は必ず建物の最外周まで振ってある）：外周と壁の位置の手がかり（2026-10-24）
    & $py -3.13 -I "$d\scripts\read_dimensions.py" $img[$fl] $cfg "$mid\$fl" | Select-Object -First 1
    & $py -3.13 -I "$d\scripts\grid_detect.py" $img[$fl] $cfg "$mid\$fl" $clean
    # 2c. 寸法の文字を読んで外周を確定（グリッドから外れた通り＝間崩れ、取りこぼした外周の壁。バルコニー等）
    #     文字の読み取りに RapidOCR を使う（入っていなければ上で自動で入れる）
    #     （入っていなければ数字は読まず、グリッド上の外周の壁だけを足す）
    & $py -3.13 -I "$d\scripts\dim_walls.py" $img[$fl] $cfg "$mid\$fl" | Select-Object -Last 1
    # 2d. 外壁のL字のすき間（玄関ドアのある壁と袖壁など）を壁でつなぐ。建具の判定より前に（11/11）
    & $py -3.13 -I "$d\scripts\close_loose_ends.py" "$mid\${fl}_walls.json" $cfg
    # 2b. 腰壁（塗りのない壁幅の帯。キッチン前・階段まわり）
    # 2a. 便器の形でトイレを見つける（文字が読めなくても。2畳以下の区画、2026-10-24）
    & $py -3.13 -I "$d\scripts\toilet_detect.py" $img[$fl] "$mid\${fl}_walls.json" "$mid\${fl}_部屋名.json" $cfg | Select-Object -Last 1
    & $py -3.13 -I "$d\scripts\koshikabe_detect.py" $img[$fl] "$mid\${fl}_walls.json" $cfg "$mid\$fl"
    # 3. 建具の判定（壁に沿って画素単位）
    & $py -3.13 -I "$d\scripts\opening_detect.py" $img[$fl] "$mid\${fl}_walls.json" $cfg "$mid\$fl"
    # 3b. 窓の記号（16020H・06011T・16011TF・11905F など）を読んで窓の種類と幅を付ける（判定結果は変えない）
    & $py -3.13 -I "$d\scripts\read_window_labels.py" $img[$fl] "$mid\${fl}_openings.json" "$mid\$fl"
    # 3c. 窓の記号がない窓は、描き方（外の扇・ずれた2枚の障子・1本の障子）で種類を見分ける
    & $py -3.13 -I "$d\scripts\window_type.py" $img[$fl] "$mid\${fl}_openings.json" $cfg "$mid\${fl}_walls.json"
    & $py -3.13 -I "$d\scripts\entrance_rule.py" "$mid\${fl}_openings.json" "$mid\${fl}_walls.json" "$mid\${fl}_部屋名.json" $cfg $img[$fl]
    & $py -3.13 -I "$d\scripts\interior_type.py" $img[$fl] "$mid\${fl}_openings.json" "$mid\${fl}_walls.json" "$mid\${fl}_部屋名.json" $cfg
    # 3d. 便器の位置と向き（make_plan が便器の部品を置く。<階>_便器_手直し.json があればそれを使う）
    #     （今まで run_all に入っていなかった。11/14）
    & $py -3.13 -I "$d\scripts\toilet_place.py" $img[$fl] "$mid\${fl}_walls.json" "$mid\${fl}_部屋名.json" $cfg "$mid\${fl}_便器.json" | Select-Object -Last 1
    # 3e. ユニットバス（浴槽の向き・ドアの壁と吊元）。make_plan が UB の部品を置き、ドアの所の壁を切る（11/16）
    & $py -3.13 -I "$d\scripts\ub_detect.py" $img[$fl] "$mid\$fl" $cfg | Select-Object -Last 1
    # 3f. キッチン（天板の位置・コンロの側・背中の側）。make_plan がキッチンの部品を置く（11/16）
    & $py -3.13 -I "$d\scripts\kitchen_detect.py" $img[$fl] "$mid\$fl" $cfg | Select-Object -Last 1
    # （洗面台の読み取り vanity_detect は区画（regions_check の結果）を使うので、下の 4. の後に走らせる）
    # 4. 区画の確認（確かめ専用）。文字を消した画像を使う。印を付けるだけで、壁・建具の結果も図面も変えない
    & $py -3.13 -I "$d\scripts\regions_check.py" $img[$fl] $clean "$mid\${fl}_walls.json" "$mid\${fl}_openings.json" "$mid\${fl}_部屋名.json" $names $cfg "$mid\$fl"
    # 4b. 洗面台（洗面・脱衣・ランドリーの区画の中の、明るい長方形。間口・背中の壁）。make_plan が洗面化粧台の部品を置く（11/16）
    & $py -3.13 -I "$d\scripts\vanity_detect.py" $img[$fl] "$mid\$fl" $cfg | Select-Object -Last 1
    # 5. 読み取りの検査：元の画像と、読み取った壁・建具・部屋を突き合わせ、読み違いの疑いを出す。確実なもの
    #    （室内の壁の窓 → ドア）は中間データで直し、直したら室内建具の種類分けをやり直す（11/15）
    #    結果は 中間\<階>_読み取りの検査.json と 出力\<階>_読み取りの検査.png（疑いの所に印）
    & $py -3.13 -I "$d\scripts\check_reading.py" $img[$fl] "$mid\$fl" $cfg --fix --out $out | Select-Object -Last 1
    $rd = Get-Content "$mid\${fl}_読み取りの検査.json" -Encoding UTF8 -Raw | ConvertFrom-Json
    if ($rd.自動修正.Count -gt 0) {
        & $py -3.13 -I "$d\scripts\interior_type.py" $img[$fl] "$mid\${fl}_openings.json" "$mid\${fl}_walls.json" "$mid\${fl}_部屋名.json" $cfg
    }
}

# 4. 壁のオフセット・柱・jww（確認画像は元の図面に重ねる。1F は上階の壁を柱位置の判断に使う）
foreach ($fl in "1F", "2F", "3F") { Remove-Item "$t\$fl.jww" -ErrorAction SilentlyContinue }  # 前の回の図を間違って写さない
# 各階は、すぐ上の階の壁を柱位置の判断に使う（最上階は上なし）
for ($i = 0; $i -lt $floors.Count; $i++) {
    $fl = $floors[$i]
    if ($i + 1 -lt $floors.Count) {
        & $py -3.13 -I "$d\scripts\make_plan.py" $planCfg "$mid\${fl}_walls.json" $img[$fl] "$t\$fl" "$mid\$($floors[$i + 1])_walls.json"
    } else {
        & $py -3.13 -I "$d\scripts\make_plan.py" $planCfg "$mid\${fl}_walls.json" $img[$fl] "$t\$fl"
    }
}

foreach ($fl in $floors) {
    Copy-Item "$t\$fl.jww" "$out\$fl.jww" -Force
    Copy-Item "$t\${fl}_壁と柱.png", "$t\${fl}_建具の配置.png" "$out\" -Force
    Copy-Item "$t\${fl}_meta.json" "$mid\" -Force
    Copy-Item "$mid\${fl}_文字の検出.png", "$mid\${fl}_壁の判定.png", "$mid\${fl}_建具の判定.png", "$mid\${fl}_区画の確認.png", "$mid\${fl}_区画から見た壁と建具.png" "$out\" -Force
}

# 採点（手直しした正解図面があるときだけ。例: 06_正解図面（採点用）\1F_正解.jww）
foreach ($fl in $floors) {
    $gt = "$d\06_正解図面（採点用）$sub\${fl}_正解.jww"
    if (Test-Path $gt) {
        "== $fl の採点"
        & $py -3.13 -I "$d\scripts\score_walls.py" $cfg "$mid\${fl}_walls.json" $gt "$mid\${fl}_meta.json"
    }
}
