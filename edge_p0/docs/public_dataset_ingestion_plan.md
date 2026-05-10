# P0 鍏紑鏁版嵁瀵煎叆涓庤浆鎹㈣鍒?
## 褰撳墠闃舵

鏈樁娈靛彧鍋氬叕寮€鏁版嵁瀵煎叆鍑嗗锛屼笉璁粌妯″瀷锛屼笉涓嬭浇澶т綋閲忔暟鎹紝涓嶆妸鍏紑浠ｇ悊鏁版嵁澶稿ぇ鎴愭渶缁?P0 鏁版嵁銆?
鏂板鑴氭湰锛?
```bash
python3 scripts/prepare_public_p0_datasets.py
```

鏂板鐩綍锛?
```text
datasets/public_p0_sources/
  README.md
  sources.json
  raw/
  converted_yolo/
  _manifests/
```

`raw/` 鍜?`converted_yolo/` 宸插湪 `.gitignore` 涓拷鐣ャ€?
## 鏁版嵁婧愪竴锛歓enodo Accessibility Barriers

鐩爣锛?
- `stairs`
- `ramp`

涓嬭浇鍚庡缓璁粨鏋勶細

```text
datasets/public_p0_sources/raw/zenodo_accessibility_barriers/
  images/
  wm_annotations.xml
```

杞崲鍛戒护锛?
```bash
python3 scripts/prepare_public_p0_datasets.py convert-cvat \
  --xml datasets/public_p0_sources/raw/zenodo_accessibility_barriers/wm_annotations.xml \
  --image-root datasets/public_p0_sources/raw/zenodo_accessibility_barriers/images \
  --output-root datasets/public_p0_yolo \
  --source-id zenodo_accessibility_barriers \
  --split train
```

榛樿鏄犲皠锛?
- `stairs -> stairs`
- `ramps/ramp -> ramp`
- `steps` 榛樿涓嶅苟鍏ワ紝闇€浜哄伐澶嶆牳銆?
濡傜‘璁?`steps` 鍙綔涓哄彴闃堕闄╂牱鏈紝鍐嶆樉寮忓惎鐢細

```bash
--include-steps-as-stairs
```

## 鏁版嵁婧愪簩锛歁endeley Obstacles Avoidance

鐩爣锛?
- `road_obstacle`

涓嬭浇骞惰В鍘嬩负 YOLO 鐩綍鍚庯紝鏍规嵁瀹為檯璺緞浼犲叆 `images-dir` 鍜?`labels-dir`锛?
```bash
python3 scripts/prepare_public_p0_datasets.py convert-yolo \
  --images-dir datasets/public_p0_sources/raw/mendeley_obstacles/images/train \
  --labels-dir datasets/public_p0_sources/raw/mendeley_obstacles/labels/train \
  --class-names pole,fence,bump,hole \
  --map pole=road_obstacle \
  --map fence=road_obstacle \
  --map bump=road_obstacle \
  --map hole=road_obstacle \
  --output-root datasets/public_p0_yolo \
  --source-id mendeley_obstacles \
  --split train
```

瑙勫垯锛?
- 鍘熷 `pole/fence/bump/hole` 缁熶竴鎶樺彔鎴?`road_obstacle`銆?- 涓嶆妸杩欎簺 source classes 鍐欏叆 P0 `data.yaml`銆?- 杞崲鍚庡繀椤绘娊鏌ュ浘鐗囧拰妗嗚川閲忋€?
## 鍐欏嚭 P0 data.yaml

```bash
python3 scripts/prepare_public_p0_datasets.py write-data-yaml \
  --output-root datasets/public_p0_yolo
```

鐢熸垚鐨勭被鍒浐瀹氫负锛?
```yaml
names:
  0: road_obstacle
  1: stairs
  2: ramp
  3: blind_road_occupied
```

## 缁熻杞崲缁撴灉

```bash
python3 scripts/prepare_public_p0_datasets.py summary \
  --root datasets/public_p0_yolo
```

## QA 瑕佹眰

杞崲鍚庡繀椤诲仛浜哄伐鎶芥煡锛?
- 绫诲埆鏄惁鍙寘鍚洓绫?P0銆?- `road_obstacle` 鏄惁鐪熺殑褰卞搷琛屼汉閫氳銆?- `stairs` 鏄惁鏍囨敞鍙伴樁鍖哄煙鏁翠綋銆?- `ramp` 鏄惁鏍囨敞鍙€氳鍧￠亾鎴栧叆鍙ｄ富浣撱€?- `blind_road_occupied` 鏄惁鏉ヨ嚜鏈湴鍏崇郴鍨嬫爣娉紝涓嶇敱瑙﹁閾鸿鏈綋鐩存帴鏄犲皠銆?- 鍥剧墖涓槸鍚︽湁闅愮鏁忔劅淇℃伅銆?
## 涓嬩竴姝?
1. 灏忔壒閲忎笅杞?Zenodo 鍜?Mendeley 鏁版嵁銆?2. 鍏堢敤 `--dry-run` 杩愯杞崲鑴氭湰銆?3. 鎶芥煡 50-100 寮犺浆鎹㈢粨鏋溿€?4. 鍐嶈繘鍏ユ寮忚浆鎹㈠拰 YOLO baseline 璁粌鍑嗗銆?