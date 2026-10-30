# protein-binding (`pbind`)

Dự đoán vị trí gắn kết ở mức residue cho 3 loại ligand: **metal**, **nuclear** (nucleic acid) và **small** (small molecule).
Đây là các notebook gốc (`legacy/notebooks/`) được tổ chức lại thành một package Python: mỗi bước là một hàm, có cache, và model final nằm gọn ở một chỗ.

Nguyên tắc giữ nguyên từ notebook gốc:

- Chọn feature/model **chỉ dựa trên DevSet1010** (5 fold chính thức của bindNode24, metric chính AUPRC). TestSet300 và TestSetNew46 chỉ dùng để xác nhận, **sau khi** đã lock.
- Tên file cache và kết quả giữ đúng layout trên Drive, nên kết quả/cache gốc dùng lại được nguyên vẹn.

## 1. Model final

Có hai profile; chi tiết (thành phần, trọng số, ngưỡng, sha1 từng file) nằm trong `data/final_model/manifest.json`.

**`final`** (chính; cần cả 4 model nền: ESMFold, ESM-2, ESM-IF, ProtT5), 80 checkpoint:

| Task | Model | Dev AUPRC | Test300 | Test46 | Test gộp | Ngưỡng |
|---|---|---|---|---|---|---|
| Metal | MLP trên ESM + ATOM + CHEM + MULTI2 + **ESM-IF** (1889 chiều), 5 fold | 0.470 | 0.386 | 0.682 | 0.409 | 0.9148 (F1) |
| Nuclear | 0.2·V1 + 0.4·V1(+ESM-IF+ProtT5, 3 seed) + 0.4·MS(+ESM-IF+ProtT5, 3 seed), 35 ckpt | 0.558 | 0.480 | 0.118 | 0.376 | 0.7079 (MCC) |
| Small | 0.1·V1 + 0.1·MS + 0.4·V1(+…, 3 seed) + 0.4·MS(+…, 3 seed), 40 ckpt | 0.487 | 0.504 | 0.472 | 0.501 | 0.4901 (MCC) |

**`lite`** (chỉ cần ESMFold + ESM-2), 25 checkpoint: metal MLP (0.459 / 0.381 / 0.685), nuclear 0.7·V1 + 0.3·MS (0.509 / 0.433 / 0.125), small 0.6·V1 + 0.4·MS (0.409 / 0.446 / 0.427).

So với lite, final tốt hơn có ý nghĩa thống kê trên test gộp ở nuclear (+0.027, CI95 [+0.009, +0.048]) và small (+0.057, [+0.044, +0.070]); metal tốt hơn có ý nghĩa trên Dev, còn trên test không có ý nghĩa (bootstrap ghép cặp theo protein, B = 10000; `notebooks/08–10`).

Weight nằm **duy nhất** trong `data/final_model/`:

```text
final_model/
├── manifest.json                           # 2 profile, trọng số, ngưỡng, nguồn gốc, sha1, số liệu
├── metal/mlp_esmif/                        # MLP__ESM__Atomic__Chem__MULTI2__ESMIF__fold{1..5}.pt   (final)
├── metal/mlp/                              # MLP__ESM__Atomic__Chem__MULTI2__fold{1..5}.pt          (lite)
├── nuclear|small/v1/, ms/                  # V1 / MS gốc: fold{1..5}.pt + fold{k}_val.npy (OOF) + fold{k}.json
└── nuclear|small/{v1,ms}_esmif_prott5_s{0,1,2}/   # V3: thêm ESMIF_EF + PROTT5, 3 seed
```

- **metal/mlp:** run gốc không lưu weight (`save_models=False`). Weight này được lấy từ lần chạy lại notebook gốc trên Colab T4 (`notebooks/07`), và 15/15 prediction trùng **từng bit** với run gốc.
- **metal/mlp_esmif:** lấy từ `notebooks/08`.
- **Nuclear/small:** V1 và MS là checkpoint gốc trên Drive; các thư mục `*_esmif_prott5_*` lấy từ `notebooks/09`, `10`.
- **Tính lại được từ đúng các file này:** Dev và test của cả hai profile, tính từ prediction đã lưu của các file trong `final_model/`, ra đúng các con số ở bảng trên.
- **Model nền:** cách dùng 4 model nền (file cố định, sha256, thiết lập chính xác) xem [docs/FOUNDATION_MODELS.md](docs/FOUNDATION_MODELS.md).
- `pbind.inference.install_final_models` tạo symlink từ thư mục pipeline (`trained_models/...`, `results/.../states/...`) tới `final_model/`. Inference và `final_models` gọi hàm này tự động, nên **không train lại** gì; mọi lần ghi chỉ thay symlink, không bao giờ ghi đè `final_model/`.
- Kiểm tra trên máy này (chỉ load, không train):
  - Dev của nuclear và small khớp tuyệt đối, vì dùng chính OOF gốc;
  - threshold khớp tới 1e-7;
  - external lệch ≤ 5e-4, metal Test300 ra 0.38144, vì ESM được tính lại trên máy.

## 2. Cấu trúc thư mục

```text
protein-binding/
├── src/pbind/                     # package
│   ├── api.py                     # API 1 hàm: cross_validate / train_final / predict (chọn feature + model bằng tên)
│   ├── paths.py, constants.py     # thư mục gốc (data/ | PBIND_BASE | Drive), hằng số
│   ├── data/                      # CSV, fold bindNode24 (bỏ 4 ID lỗi), cấu trúc PDB (chain A khớp sequence)
│   ├── features/                  # @feature_group: ESM-2, AA/SPATIAL/LOCAL, ATOM/CHEM, MECH1, MULTI2, SEQCTX, NA_*, SM_*
│   │   └── store.py               # FEATURE STORE: tính 1 lần, mỗi block 1 file cột
│   ├── groups/                    # patch (nuclear) / pocket (small) cho hypergraph
│   ├── models/                    # @register_model: residue MLP/KAN, fusion, hypergraph; losses
│   ├── training/                  # trainer GEAR (MLP/KAN/XGBoost), CV local, CV hypergraph
│   ├── evaluation/, reporting/    # metric, threshold (chỉ chọn trên Dev OOF), ensemble, báo cáo
│   ├── tasks/                     # TaskSpec: grid, tên, thư mục, lock gốc và số gốc của từng task
│   ├── backend/                   # native.py (mặc định, bản port) | gear.py (gói gear_ultimate trên Drive)
│   ├── pipelines/                 # mỗi cell notebook = 1 hàm (screens, v1, v2, final, metal, crosstask, ...)
│   └── inference/                 # final.py (final_model/), metal.py (inference metal)
├── scripts/                       # check_setup (chạy đầu tiên), precompute_features, train_metal, infer_metal,
│                                  # run_pipeline, check_feature_parity, build_dna_rna_labels, verify_installation
├── notebooks/                     # notebook mỏng chỉ gọi hàm (00–05) + notebook Colab (07–15)
├── tests/                         # pytest trên dữ liệu giả (không cần Drive)
├── legacy/
│   ├── notebooks/                 # notebook gốc (kèm output), gồm protein_folding_ESMFold_v1.ipynb
│   └── gear_ultimate_backend/     # zip mã nguồn backend gốc (chỉ để đối chiếu)
└── data/                          # không vào git; cùng layout với Drive Protein-binding-Phi/
    ├── data/development_set/      # train CSV, TestSet300 CSV, ids_split1..5.txt
    ├── data/independent_set/      # TestSetNew46 CSV
    ├── esmFold_pdb_files/         # cấu trúc ESMFold v1 có sẵn
    ├── final_model/               # 3 model final (mục 1)
    ├── feature_store/             # feature đã tính (mục 4)
    ├── cache/                     # cache theo họ feature, PDB chain A, patch/pocket bank
    ├── results/, trained_models/  # kết quả pipeline (+ symlink tới final_model/)
    ├── external_benchmarks/       # bộ GraphBind (notebook 14) kèm PROVENANCE.txt
    └── _local/                    # scratch nhanh (PDB chain A đã chuẩn hoá)
```

| Notebook gốc | Notebook mới | Code |
|---|---|---|
| `ALL_BINDINGS_DATASET_ANATOMY_DEV1010_300_46` | `00_dataset_anatomy` | `analysis/anatomy.py` |
| `GEAR_ULTIMATE_METAL_BINDING`, `GEAR_METAL_ONLY_ULTIMATE_V2` | `01_metal`, `05_metal_inference`, `07` | `pipelines/metal.py`, `tasks/metal.py`, `inference/metal.py` |
| `NUCLEAR_ULTIMATE_V1_DATA_DRIVEN_PATCH` | `02_nuclear` | `features/nuclear.py`, `groups/nuclear_patches.py`, `pipelines/{screens,v1,v2,final}.py` |
| `SMALL_ULTIMATE_V1_DATA_DRIVEN_POCKET` | `03_small` | `features/small.py`, `groups/small_pockets.py`, cùng các pipeline trên |
| `NUCLEAR/SMALL_STANDALONE_*_ADDIN` | `04_crosstask_evo` | `pipelines/crosstask.py`, `features/derived.py` |
| `GEAR_METAL_V4_FULL_INSIGHT_HYPERGRAPH` (chưa từng chạy xong) | `08` (đã sửa) | — |
| — | `09`, `10` (V3 nuclear / small) | `pipelines/v3`, cùng `final.py` |
| — | `12`, `13`, `15` (dòng DNA / RNA) | `scripts/build_dna_rna_labels.py` |

## 3. Cài đặt và data

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[all,dev]"
```

Extra: `esm` (transformers), `structure` (mdtraj), `xgboost`, `kan` (pykan); `all` = 3 cái đầu. XGBoost/KAN chỉ dùng trong screen metal, model final không cần. Máy này PyPI rất chậm: cài `-e ".[esm,structure,dev]" --offline` từ cache của uv (xgboost chưa có trong cache).

**Môi trường conda cho thực nghiệm và vẽ hình:** `AdaBind` (`~/miniconda3/envs/AdaBind`, Python 3.11: numpy, pandas, scipy, matplotlib, biopython, pymol-open-source). Mọi code chạy thực nghiệm hoặc phát triển thêm dùng env này (`conda activate AdaBind`); cần torch/transformers thì cài thêm vào chính env này, không tạo env khác.

Data tải từ Drive `Protein-binding-Phi/` vào `data/`, **giữ nguyên layout**. Thư mục gốc được chọn theo thứ tự: `base=` → biến `PBIND_BASE` → `<repo>/data/` → Drive. Chép vào `data/` là xong, không cần cấu hình gì.

```text
data/                                  <- hoặc trỏ PBIND_BASE vào thư mục khác có cùng layout
├── data/
│   ├── development_set/
│   │   ├── full_grouped_train_binding_sites_df.csv     BẮT BUỘC  nhãn DevSet1010 (1014 dòng, dùng 1010)
│   │   ├── full_grouped_test_binding_sites_df.csv      BẮT BUỘC  nhãn TestSet300
│   │   ├── ids_split1..5.txt                           BẮT BUỘC  fold chính thức của bindNode24
│   │   └── nuclear_dna_rna_labels.csv                            lớp DNA/RNA từ BioLiP2 (notebook 12/13/15)
│   └── independent_set/
│       └── grouped_test_46_new_binding_sites.csv       BẮT BUỘC  nhãn TestSetNew46
├── esmFold_pdb_files/                 711 MB  BẮT BUỘC  cấu trúc, mọi feature cấu trúc dựng từ đây
├── final_model/                       291 MB  cho inference  metal/ nuclear/ small/ + manifest.json
├── feature_store/                     1.3 GB  dựng lại được bằng scripts/precompute_features.py
├── cache/                             1.8 GB  cache GEAR, dựng lại được
├── baseline_reproduction_master_v6/           dự đoán bindEmbed21DL / bindNode24, chỉ để vẽ bảng so sánh
└── results/                                   đầu ra, tự sinh
```

Chỉ **5 mục BẮT BUỘC** (3 CSV + `ids_split1..5.txt` + `esmFold_pdb_files/`) là đủ để train lại từ đầu; thêm `final_model/` là chạy được inference. `feature_store/` và `cache/` chỉ để khỏi tính lại, thiếu thì tự dựng.

Kiểm trước khi chạy bất cứ thứ gì:

```bash
.venv/bin/python scripts/check_setup.py        # exit 0 = đủ input; in ra cái nào thiếu và ở đường dẫn nào
```

### Model nền: tải gì, bỏ đâu

Repo **chỉ có code**. Không file weight nào, không file data nào được commit. Có hai đường dùng, cần những thứ khác nhau:

**A. Dựng feature store / train lại** — chỉ cần **ESM-2 650M**.

ESM-2 là model nền duy nhất mà feature store cần (block `ESM`, 1 trong 29 block). Nó **tự tải** từ HuggingFace ở lần chạy đầu (~2.6 GB), không phải đặt tay. Máy không có mạng thì tải trước rồi trỏ đường dẫn:

```bash
export PBIND_ESM_MODEL=/duong/dan/esm2_t33_650M_UR50D     # mặc định: facebook/esm2_t33_650M_UR50D trên HF
.venv/bin/python scripts/precompute_features.py
```

ESMFold **không cần** — cấu trúc đã có sẵn trong `esmFold_pdb_files/`, chỉ fold khi thêm protein mới. ProtT5 và ESM-IF **không cần** — chúng không phải block của store (xem "Package dựng lại được tới đâu" ở mục 5).

**B. Inference protein mới bằng `pbind.Predictor`** — cần đủ **cả bốn** model nền, đặt tay, offline:

```text
model_weights/                        <- truyền vào Predictor(..., weights_dir=...)
├── MANIFEST.json                     nguồn + revision + sha256 của từng file
├── esmfold_v1/                       8.44 GB   facebook/esmfold_v1 @ 75a3841e
├── esm2_t33_650M_UR50D/              2.61 GB   facebook/esm2_t33_650M_UR50D @ 08e4846e
├── prot_t5_xl_half_uniref50-enc/     2.42 GB   Rostlab/prot_t5_xl_half_uniref50-enc @ 94a6abc0
└── esm_if1/                          1.70 GB   esm_if1_gvp4_t16_142M_UR50.pt (fair-esm)
```

Tên thư mục phải **đúng từng ký tự** — `predictor.py` tra theo tên này. Revision và sha256 đầy đủ trong [`docs/FOUNDATION_MODELS.md`](docs/FOUNDATION_MODELS.md); kiểm bằng `scripts/verify_installation.py --weights-dir ...`.

### Thứ tự làm

Tải các file nén về **một thư mục bất kỳ**, rồi để script tự đặt vào đúng chỗ:

```bash
bash scripts/setup_data.sh ~/Downloads                  # xem trước: thêm --dry-run
```

| file tải về | tạo ra | |
|---|---|---|
| `pbind_data.zip` | `data/data/` | **bắt buộc** — 3 CSV + `ids_split1..5.txt` |
| `pbind_esmfold_pdb.zip` | `data/esmFold_pdb_files/` | **bắt buộc** — cấu trúc ESMFold |
| `pbind_final_model.zip` | `data/final_model/` | 80 checkpoint, cần khi muốn inference luôn mà không train |
| `pbind_feature_store.zip` | `data/feature_store/` | tuỳ chọn — 29 block đã tính sẵn, khỏi chạy lại |
| `pbind_cache.zip` | `data/cache/` | tuỳ chọn |
| `foundation_models.zip` | `model_weights/` | chỉ khi inference protein mới (đường B ở trên) |

Mỗi file nén phải chứa **đúng thư mục đích ở gốc** (ví dụ `pbind_final_model.zip` mở ra là `final_model/`). Script đọc tên thư mục gốc trước khi giải nén, sai là báo lỗi và bỏ qua chứ không đổ nhầm chỗ. Chạy lại thì bỏ qua cái đã có, muốn ghi đè thì thêm `--force`. Đặt ở nơi khác thì `--base /duong/dan`. Giải nén xong nó tự gọi `check_setup.py`.

Rồi:

```bash
# 1. dựng feature store (29 block; ESM-2 tự tải lần đầu, ~2.6 GB)
.venv/bin/python scripts/precompute_features.py
# 2a. chỉ muốn chạy inference bằng model có sẵn -> xong rồi, không cần train
.venv/bin/python scripts/infer_metal.py --split TestSet300
# 2b. muốn train lại
.venv/bin/python scripts/train_metal.py                      # metal, end to end
.venv/bin/python scripts/run_pipeline.py nuclear --stages all
```

Nếu đã tải `pbind_feature_store.zip` thì bỏ qua bước 1. Nhớ giới hạn ở mục 5: train lại bằng package **không** dựng lại được 80% trọng số của nuclear/small và toàn bộ metal final — phần đó nằm trong `pbind_final_model.zip`.

## 4. Feature store

Mỗi nhóm feature là **một hàm nhận một protein**, đăng ký bằng `@feature_group`. Feature store gọi các hàm này **một lần** cho mỗi protein, rồi lưu mỗi block thành một file cột:

```text
feature_store/
├── manifest.json                      # block -> dim, dtype, nhóm, tên từng cột
└── <DevSet1010|TestSet300|TestSetNew46>/
    ├── index.csv  labels.npz          # prot_id, start, end, sequence | nhãn metal/nuclear/small (int8)
    └── <BLOCK>.npy                    # 29 block, 1997 cột (ESM float16, còn lại float32)
```

```python
from pbind.features.store import FeatureStore
fs = FeatureStore("data/feature_store")
X  = fs.matrix("TestSet300", ["ESM", "ATOM", "CHEM"])           # (62689, 1319)
df = fs.frame("TestSet300", ["ATOM"], labels=["metal_binding"])  # bảng có tên cột
```

```bash
.venv/bin/python scripts/precompute_features.py              # mọi nhóm + patch/pocket bank (chỉ tính phần còn thiếu)
.venv/bin/python scripts/precompute_features.py --store-only --force   # dựng lại store từ cache GỐC chép từ Drive
```

- **Nguồn cấu trúc:** nhóm GEAR tính từ file ESMFold; nhóm NA_*/SM_* tính từ bản chain A khớp sequence (đúng như notebook gốc).
- **Dùng lại cache gốc:** nếu `cache/spatial_revamp/*.pkl` và cache NA/SM theo từng protein có sẵn (ví dụ chép từ Drive) thì được **nhập thẳng** vào store. `require_legacy=True` bắt buộc nhập 100% từ cache gốc, dùng khi cần tái lập khít.
- **An toàn hàng:** mỗi block phải có đúng 1 hàng cho mỗi residue; block thiếu hàng sẽ báo lỗi chứ không âm thầm lệch.

## 5. Chạy

```bash
# inference metal trên 2 bộ test bằng model final (không train)
.venv/bin/python scripts/infer_metal.py --split TestSet300
# nuclear / small: ensemble V1+MS từ checkpoint gốc, báo cáo + so với số gốc (FINAL_reproduction_vs_original.csv)
.venv/bin/python scripts/run_pipeline.py nuclear --stages final_models
.venv/bin/python scripts/run_pipeline.py small --stages final_models
# chạy lại toàn bộ thí nghiệm của một task (screen, ladder, hypergraph, V2, ...), có resume
.venv/bin/python scripts/run_pipeline.py nuclear --stages all
```

**API một hàm** (`pbind/api.py`): chọn feature và model bằng tên. Mỗi lời gọi đi đúng runner cũ (cùng cache, seed, file fold), nên MLP qua API ra trùng từng bit với pipeline.

```python
from pbind import api
ws = api.workspace("metal")
api.list_features(); api.list_models()
api.cross_validate(ws, model="MLP", features="ESM+Atomic+Chem+MULTI2")          # chỉ Dev
api.train_final(ws, model="MLP", features="ESM+MECH1", neural_epochs=20)        # lưu weight + đánh giá 300/46
api.predict(ws, model="NA_PatchHG_Attention", features="...", split="TestSet300")
```

**Thêm feature:** viết một hàm `@feature_group("TEN", columns={...}, needs="sequence"|"structure")`; store tự tính, và hàm dùng được ngay ở `features="ESM+TEN"`.
**Thêm model:** viết một hàm `@api.register_model("TEN", kind="residue"|"fusion"|"group")`.
Chữ ký của từng loại xem trong `features/api.py` và `models/registry.py`.

### Package dựng lại được tới đâu

Feature store có **31 block**. `PROTT5` và `ESMIF_EF` — hai embedding mà tầng V3 cần — **đã được đăng ký `@feature_group`**, nên `precompute_features.py` tính được chúng:

| block | dim | dtype | tính bằng | cache gốc đọc lại được |
|---|---|---|---|---|
| `PROTT5` | 1024 | fp16 | `features/prott5.py`, env chính | `cache/prott5_xl_half_v1/<split>/<pid>.npy` |
| `ESMIF_EF` | 512 | fp16 | `features/esmif.py`, **cần env fair-esm riêng** | `cache/gear_metal_v3/esmif_esmfold_v1/<split>/esm_if/<pid>.npy` |

Weight tìm theo thứ tự `PBIND_PROTT5_MODEL` / `PBIND_ESMIF_WEIGHTS` → `model_weights/<tên>/` → `data/foundation_models/<tên>/` → tải từ HuggingFace. Giải nén `foundation_models.zip` vào `model_weights/` là chạy được ngay.

Hai block này được đánh dấu **`optional=True`**, nên `precompute_features.py` chạy trần **không** đụng tới chúng — máy không có weight hay không có env fair-esm vẫn dựng được 29 block kia bình thường. Muốn tính thì gọi tên:

```bash
.venv/bin/python scripts/precompute_features.py --groups PROTT5
.venv/bin/python scripts/precompute_features.py --groups ESMIF_EF    # chạy trong env fair-esm
```

Lưu ý có **hai cache ESM-IF khác nhau, không thay thế nhau được**: `ESMIF` (float32, `esmif_v1/`, dùng cho screen metal, gắn qua `backend/gear.py`) và `ESMIF_EF` (float16, `esmif_esmfold_v1/`, đúng cái V3 đã train).

**Nhưng vẫn chưa tự dựng lại được 80 checkpoint**, vì còn thiếu **tầng V3** trong `pipelines/` (bagging 3 seed × 2 kiến trúc, tìm trọng số ensemble cross-fit trên Dev). Hiện trạng:

| member | trọng số trong `final` | dựng lại bằng `src/pbind`? |
|---|---|---|
| `metal/mlp_esmif` | 1.0 | **không** — notebook 08, cần ESM-IF |
| `nuclear/v1` | 0.2 | có — `run_pipeline.py nuclear --stages hypergraph` |
| `nuclear/{v1,ms}_esmif_prott5_s0..2` | 0.8 | **không** — notebook 09, cần ESM-IF + ProtT5 |
| `small/v1`, `small/ms` | 0.2 | có — `run_pipeline.py small --stages hypergraph ms` |
| `small/{v1,ms}_esmif_prott5_s0..2` | 0.8 | **không** — notebook 10 |

Nói gọn: **80% trọng số của nuclear và small, và toàn bộ metal final, hiện vẫn đến từ notebook.** `run_pipeline.py <task> --stages final_models` **không train** — nó link checkpoint gốc trong `final_model/<task>/` vào thư mục pipeline rồi dùng lại, đúng như docstring của `pipelines/final.py:final_models` ghi.

Đường dựng lại được bằng package: `scripts/precompute_features.py` (31 block, gồm cả `PROTT5` và `ESMIF_EF`) → `scripts/train_metal.py` (metal MLP, profile `lite`) hoặc `run_pipeline.py <task> --stages all` (V1/MS của nuclear/small).

**Việc còn lại để khép kín:** port tầng V3 từ cell V3-4…V3-7 của notebook 09/10 thành một hàm trong `pipelines/`, rồi thêm stage `v3` vào `TASK_STAGES` của `scripts/run_pipeline.py`. Feature đã sẵn sàng; chỉ còn phần điều phối huấn luyện. Nguồn gốc từng member ghi trong `final_model/manifest.json`, khoá `sources`.

## 6. Notebook Colab

- `notebooks/07_METAL_FINAL_save_weights_colab.ipynb`: **đã chạy xong**. Đây là notebook V2 gốc, thêm `save_models=True`; kết quả `BIT-EXACT` 15/15, weight chính là `final_model/metal/mlp`.
- `notebooks/08_METAL_V4_fixed_colab.ipynb`: thí nghiệm V4 (ESM-IF, SITECTX32, fusion, graph, hypergraph site kim loại). Các chỗ sửa `[V4-FIX]` so với bản gốc:
  - thay `deepcopy` O(L²) (lý do V4 chưa từng chạy xong);
  - dùng cùng cấu trúc ESMFold cho mọi split;
  - cache gắn với nguồn cấu trúc;
  - đưa model final hiện tại vào lock, nên V4 chỉ thay model final khi hơn ≥ 0.002 trên Dev.
- `notebooks/09_NUCLEAR_V3_improve_colab.ipynb`, `notebooks/10_SMALL_V3_improve_colab.ipynb`: V3 cho nuclear và small — thêm ESM-IF + ProtT5, bagging 3 seed, tìm trọng số ensemble **cross-fit trên Dev**, bootstrap ghép cặp theo protein B=10000. Kết quả là `final_model/nuclear` và `final_model/small`.
- `notebooks/11_NUCLEAR_DNA_RNA_colab.ipynb`: thử nghiệm đa nhiệm DNA/RNA như nhánh phụ của nuclear. **Bị thay thế** bởi 12/13.
- `notebooks/12_DNA_model_line_colab.ipynb`, `notebooks/13_RNA_model_line_colab.ipynb`: hai dòng model riêng cho DNA và RNA. Cùng kiến trúc với nuclear (V1 + MS hypergraph, cùng modality), **chỉ đổi đích**. Nhãn lớp ligand lấy từ BioLiP2 qua `scripts/build_dna_rna_labels.py`; 6 protein Dev không tra được thì bị mask khỏi loss và mọi metric.
- `notebooks/15_DNA_RNA_evaluation_colab.ipynb`: chạy **trong cùng session** sau 12 và 13. Chấm DNA và RNA như **hai bài toán độc lập**, đúng cách metal/small được chấm, rồi đóng gói bộ inference.
- `notebooks/14_EXTERNAL_BENCHMARK_colab.ipynb`: benchmark GraphBind (`DNA-129_Test`, `RNA-117_Test`). **Không import gì từ dự án**, tự tải dữ liệu, chạy được trong Colab trắng. Kết luận hiện tại là **âm tính** — xem mục 8.

## 7. Inference protein mới

1. **Fold bằng ESMFold v1 đúng cấu hình gốc** (`legacy/notebooks/protein_folding_ESMFold_v1.ipynb`): HF `facebook/esmfold_v1`, `model.esm.half()`, `trunk.set_chunk_size(64)`, TF32, recycle mặc định, `to_pdb` của openfold (B-factor = pLDDT thang 0–1). Cần GPU ≥ 16GB. Miễn phí thì dùng API ESM Atlas (≤ 400 aa) hoặc Colab/Kaggle chạy theo lô.
2. **Không fold sequence có `X`/`U`.** Vòng lặp gốc dùng `try/except/finally`; khi fold lỗi, nó ghi cấu trúc của **protein trước** dưới tên protein hiện tại. Q00277 và Q9NZV6 bị đúng lỗi này; C8BD48 và P84801 thì mất residue `X`. Đó là 4 protein bị loại khỏi Dev (1014 → 1010, trùng với split bindNode24).
3. **Embedding ESM-2 650M** (trung bình 2 layer cuối, bỏ BOS/EOS), sau đó tính feature và chạy 3 model như mục 1.

## 8. Ghi chú

- **Tất định:** train là tất định trên cùng GPU và cùng phiên bản phần mềm. Đổi GPU hoặc torch thì AUPRC từng fold có thể lệch khoảng ±0.01; muốn khít với run gốc thì dùng Colab T4 với torch 2.11 và cache gốc.
- **Không ghi đè kết quả cũ:** chạy lại trên thư mục run gốc với config khác thì kết quả cũ được chuyển vào `superseded/<hash>/`.
- **Block CONF:** code gốc chia pLDDT cho 100, nhưng ESMFold lưu pLDDT thang 0–1, nên CONF ra rất nhỏ. Giữ y như gốc; không model final nào dùng CONF.
- **Cache metal:** cache ESM, structure, ATOM/CHEM, MECH1/MULTI2 trong `cache/spatial_revamp/` là bản tính lại trên máy. Muốn feature khít tuyệt đối thì chép bản gốc từ Drive vào rồi chạy `precompute_features.py --store-only --force`, hoặc so trước bằng `scripts/check_feature_parity.py`.
- **DNA/RNA chỉ báo cáo như hai bài toán riêng.** Khi tách, lớp ligand nằm trong **nhãn**, không nằm trong đầu vào — mọi model vẫn chấm cả 300/46 protein nên so sánh công bằng. Ngược lại, **gộp hai model chuyên trở lại thành một dự đoán nuclear là oracle** và không được báo cáo: route một protein sang model DNA tự nó đã là tuyên bố "protein này gắn DNA", mà 1470/1470 residue dương nằm gọn trong 66 trên 300 protein. Đối chứng: cộng hằng số 0.2 vào điểm nuclear của 56 protein DNA, **không dùng model chuyên nào**, đạt 0.5348 — ăn đứt bản route thật 0.5273. Dòng RNA cũng yếu: cận dưới CI chỉ +0.0016 trên Test300 và thua trên Test46.
- **Benchmark GraphBind: không đánh giá được.** `DNA-129_Test` chồng lấn với DevSet1010 — `5wx9_A` trùng tuyệt đối `Q9LSX0`; `5d2s_A`, `5uk7_H`, `6c31_A` là chuỗi con của `Q5FBS0`, `C1IFD2`, `O53623`; `5jrc_C` chứa `Q74ML9`. Chạy model huấn luyện trên DevSet1010 lên đó là nhiễm train-test. Notebook 14 có cổng chặn cứng. Muốn thi ở sân đó thì phải huấn luyện lại từ đầu trên `DNA-573_Train` / `RNA-495_Train`. Mật độ dương của họ (~6%) cũng gấp ba của mình (2.08%) nên AUPRC hai bên vốn không đặt cạnh nhau được.

## 9. Test

```bash
.venv/bin/python -m pytest -q tests/
```
