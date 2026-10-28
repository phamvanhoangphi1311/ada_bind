# Tích hợp model dự đoán binding site vào ứng dụng

Tài liệu cho người làm ứng dụng. Ứng dụng nhận **một chuỗi protein + một loại binding** và trả về **vị trí các residue binding** của loại đó.

Tài liệu mô tả:
- phần backend cần dựng;
- cách dùng các file model đã có trên máy;
- hợp đồng API giữa UI và backend, để làm UI ngay từ bây giờ.

Đọc kèm `FOUNDATION_MODELS.md`: thiết lập chính xác của từng model nền.

---

## 1. Luồng xử lý

| Loại binding (`binding_type`) | Ý nghĩa |
|---|---|
| `metal` | gắn ion kim loại (Zn, Ca, Mg, Fe, …) |
| `nuclear` | gắn DNA/RNA |
| `small` | gắn phân tử nhỏ (thuốc, cofactor, …) |

```
UI: sequence + binding_type
 │
 ▼
Backend (Python)
 1. ESMFold — GỌI API ──▶ PDB (cấu trúc 3D) + pLDDT (độ tin cậy cấu trúc, thang 0–1)
 2. Kiểm tra + chuẩn hoá PDB về chain A trùng khít sequence
 3. Embedding: ESM-2 (mọi loại) · ESM-IF (mọi loại) · ProtT5 (chỉ nuclear, small)
 4. Feature viết tay của loại binding đó (tính từ PDB)
 5. Chạy checkpoint của loại đó → xác suất cho từng residue → so với ngưỡng
 │
 ▼
UI: xác suất từng residue + danh sách residue binding
```

Mỗi yêu cầu **chỉ chạy model của loại binding được chọn**:

| `binding_type` | Model nền cần | Checkpoint (`final`) | Ngưỡng |
|---|---|---|---|
| `metal` | ESMFold (API), ESM-2, ESM-IF | 5 (`metal\mlp_esmif`) | 0.9148 |
| `nuclear` | ESMFold (API), ESM-2, ESM-IF, ProtT5 | 35 | 0.7079 |
| `small` | ESMFold (API), ESM-2, ESM-IF, ProtT5 | 40 | 0.4901 |

Ngưỡng và danh sách checkpoint lấy từ `manifest.json` (`profiles.final.<binding_type>`), không viết cứng trong code.

## 2. ESMFold qua API

**API công khai (ESM Atlas):**

```
POST https://api.esmatlas.com/foldSequence/v1/pdb/
body: chuỗi sequence (text thuần, không JSON)
→ 200: file PDB (text)     → 413: "Sequence is longer than 400."
```

Đã kiểm tra trực tiếp:
- **Cùng model:** API trả về "ESMFOLD V1 PREDICTION", chain A, pLDDT thang 0–1, giống dữ liệu train.
- **Cấu trúc khớp với lúc train:** fold lại 3 protein của bộ test qua API rồi so với PDB lúc train:
  - D0VX23 (pLDDT 0.94): RMSD **0.001 Å**;
  - P00630 (pLDDT 0.80): RMSD **0.06 Å**;
  - Q00459 (pLDDT chỉ 0.44): RMSD **2.1 Å**. Vùng ESMFold không chắc chắn vốn không ổn định giữa các máy.
- **Thời gian:** 7–25 giây mỗi protein (100–330 aa).

**Giới hạn và bắt buộc:**
1. **Tối đa 400 aa.** Dữ liệu train có protein dài tới 813 aa. Muốn hỗ trợ chuỗi > 400 thì phải có ESMFold tự host (GPU ≥ 16 GB, đúng thiết lập ở mục 2 của `FOUNDATION_MODELS.md`) làm đường dự phòng. Không có thì UI phải chặn > 400 với thông báo rõ ràng.
2. **API lỗi hoặc timeout thì báo lỗi.** Không dùng lại PDB cũ, không dùng PDB của protein khác. Vòng fold gốc từng mắc đúng lỗi này.
3. **Kiểm tra PDB trả về:** chain A phải có đúng `len(sequence)` residue, và đọc ra phải trùng sequence. Không trùng thì báo lỗi.
4. **pLDDT chỉ dùng để hiển thị.** Không feature nào của model final dùng pLDDT, nên thang của nó không ảnh hưởng dự đoán.
5. **Cache PDB** theo `sha1(sequence)`, vì API chậm và có thể giới hạn số lần gọi.

## 3. File trên máy chạy backend

### 3.1 Model nền: `D:\Protein_Ligand_Binding\model_weights\`

```
model_weights\
├── esm2_t33_650M_UR50D\            (2.61 GB)   mọi loại binding
├── esm_if1\                        (1.70 GB)   mọi loại binding
├── prot_t5_xl_half_uniref50-enc\   (2.42 GB)   nuclear, small
└── MANIFEST.json                   nguồn + revision + sha256
```

- Không cần `esmfold_v1` khi fold qua API, trừ khi muốn tự host làm đường dự phòng cho chuỗi > 400 aa.
- `FOUNDATION_MODELS.md` gọi thư mục này là `foundation_models/`.
- `model_weights\final_model\` **không thuộc chỗ này.** Nếu là bản sao thì xoá, chỉ giữ một bản ở mục 3.2, để tránh dùng nhầm phiên bản.
- **Kiểm tra sau khi tải:** chạy trong `model_weights\`. Mọi dòng phải ra `OK` (dòng `esmfold_v1` báo thiếu file thì bỏ qua nếu không tự host).

```python
import json, hashlib, pathlib
for name, e in json.load(open("MANIFEST.json")).items():
    for f, want in e["sha256"].items():
        fp = pathlib.Path(name) / f
        if not fp.exists():
            print(name, f, "MISSING"); continue
        h = hashlib.sha256()
        with open(fp, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""): h.update(b)
        print(name, f, "OK" if h.hexdigest() == want else "MISMATCH")
```

### 3.2 Model đã train: `D:\Protein_Ligand_Binding\final_models\final_model\`

```
final_model\
├── manifest.json                   ← cấu hình: checkpoint nào, trọng số, ngưỡng, checksum. KHÔNG sửa.
├── metal\mlp_esmif\                5    metal (final)
├── nuclear\v1\, nuclear\{v1,ms}_esmif_prott5_s{0,1,2}\      35   nuclear (final)
├── small\v1\, ms\, small\{v1,ms}_esmif_prott5_s{0,1,2}\     40   small (final)
└── metal\mlp\, nuclear\ms\         chỉ dùng cho profile lite
```

- Giữ nguyên tên thư mục và tên file.
- **Kiểm tra:** chạy trong `final_model\`. Phải in `0 mismatches`.

```python
import json, hashlib, pathlib
man = json.load(open("manifest.json")); bad = 0
for member, info in man["members"].items():
    for f, want in info["sha1_16"].items():
        got = hashlib.sha1((pathlib.Path(member) / f).read_bytes()).hexdigest()[:16]
        bad += got != want
print(bad, "mismatches")
```

### 3.3 Profile

- Mặc định dùng **`final`**.
- **`lite`** chỉ cần ESM-2 (không cần ESM-IF, ProtT5), dùng 25 checkpoint, kém hơn một chút: Test300 AUPRC metal 0.381 so với 0.386, nuclear 0.433 so với 0.480, small 0.446 so với 0.504.
- `lite` chỉ để dự phòng khi chưa dựng được môi trường ESM-IF.

## 4. Phần cứng và môi trường backend

- **GPU NVIDIA khuyến nghị ≥ 12 GB VRAM.** Ba model nền tốn khoảng 6–7 GB, ESMFold chạy ở API. Chạy CPU vẫn được nhưng chậm, nhất là ProtT5-XL.
- **RAM ≥ 16 GB. Ổ đĩa** khoảng 7 GB cho weight, cộng 0.3 GB cho final model.
- **Hệ điều hành:** nên chạy backend trên **Linux hoặc WSL2 (Ubuntu) trên Windows**. Mọi thứ đã được kiểm chứng trên Linux; `torch-scatter`, `torch-geometric` (cho ESM-IF) và `mdtraj` cài trên Windows gốc dễ lỗi. UI chạy ở đâu cũng được.
- **Hai môi trường Python** (chi tiết trong `FOUNDATION_MODELS.md`):
  1. **Môi trường chính:** ESM-2, ProtT5, feature, checkpoint, web server.
  2. **Môi trường ESM-IF riêng:** Python 3.10, torch 2.4.1, fair-esm 2.0.0. Backend gọi qua subprocess hoặc một service nhỏ.
- **Load một lần:** model nền và checkpoint load khi khởi động server, không load lại mỗi request.
- **Chạy offline** (trừ ESMFold API): đặt `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, load model từ đường dẫn local, và chép file ESM-IF vào `%TORCH_HOME%\hub\checkpoints\`. Nếu không đặt `TORCH_HOME` thì là `%USERPROFILE%\.cache\torch\hub\checkpoints\`; trên Linux/WSL2 là `~/.cache/torch/hub/checkpoints/`.

## 5. Hợp đồng API (UI ↔ backend)

```
POST /predict
body: {"sequence": "MKTAYIAKQR...", "binding_type": "metal"}      binding_type ∈ metal | nuclear | small
→ 200 result (bên dưới)
→ 400 sequence không hợp lệ | 413 dài hơn mức hỗ trợ | 502 ESMFold API lỗi (thử lại sau)

GET /health → {"status": "ok", "model_version": "<sha1 của manifest.json>", "max_length": 400}
```

Thời gian một request: ESMFold API mất 7–25 giây, cộng vài giây xử lý. UI nên có trạng thái "đang dự đoán" và timeout khoảng 120 giây.

**Kiểm tra input** (backend trả 400 nếu không đạt):
- chỉ gồm 20 chữ `ACDEFGHIKLMNPQRSTVWY`; đổi sang chữ hoa, bỏ khoảng trắng và dòng header FASTA;
- độ dài từ 30 đến 400. Dài hơn 400 thì cần ESMFold tự host (mục 2).

**Kết quả:**

```json
{
  "sequence": "MKTAYIAKQR...",
  "length": 245,
  "binding_type": "metal",
  "profile": "final",
  "model_version": "3f2a9c...",
  "threshold": 0.9148,
  "probability": [0.002, 0.001, 0.97, ...],
  "binding": [false, false, true, ...],
  "binding_residues": [{"index": 3, "aa": "T", "probability": 0.97}],
  "structure": {"source": "esmatlas", "plddt": [0.91, 0.88, ...], "pdb": "ATOM ..."}
}
```

- `index` đánh số từ 1 theo sequence.
- `probability`, `binding`, `plddt` dài đúng bằng `length`. `binding[i] = probability[i] >= threshold`.
- Cùng `sequence` + `binding_type` + `model_version` thì kết quả không đổi, nên có thể cache.

Có thể làm UI ngay với dữ liệu giả theo đúng format trên.

## 6. Gợi ý hiển thị

- **Hai lớp thông tin:** tô màu từng residue theo `probability`; đánh dấu các residue có `binding = true`. Nếu có viewer 3D thì tô trên cấu trúc `pdb`.
- **Độ tin cậy cấu trúc:** hiển thị pLDDT. Vùng pLDDT < 0.7 thì cấu trúc, và dự đoán dựa trên cấu trúc ở đó, kém tin cậy.
- **Không so xác suất giữa các loại binding.** Mỗi loại có thang và ngưỡng riêng; metal có ngưỡng cao nên thường chỉ ít residue được đánh dấu.
- **Độ chính xác thực tế, nên ghi ở phần giới thiệu.** Đo trên 2 tập test độc lập, tại ngưỡng của profile `final`:

  | Loại | Trong các residue được đánh dấu, tỉ lệ đúng (precision) | Trong các residue binding thật, tỉ lệ bắt được (recall) |
  |---|---|---|
  | metal | 66–79% | 36–64% (thận trọng: ít đánh dấu nhưng phần lớn đúng) |
  | nuclear | 67% (Test300); **12% (Test46)** | 23–33% |
  | small | 47–49% | 50–55% |

  Nuclear trên tập protein mới nhất (Test46) kém rõ, kể cả các phương pháp đã công bố cũng vậy. Đây là công cụ gợi ý cho nghiên cứu, không thay thế thực nghiệm.

## 7. Phần nào ai làm

| Việc | Ai làm | Trạng thái |
|---|---|---|
| UI, endpoint `/predict`, gọi ESMFold API, cache, xử lý lỗi | Người làm ứng dụng | Làm được ngay theo mục 2 và 5 |
| Dựng 2 môi trường Python trên máy backend | Người làm ứng dụng, theo `FOUNDATION_MODELS.md` | Làm được ngay |
| Hàm `pbind.predict(...)`: nhận sequence + binding_type + PDB, trả đúng `result` ở mục 5 | Nhóm model | **Đang hoàn thiện** |
| Golden set + `verify_installation.py`: chạy lại các protein mẫu, báo PASS/FAIL so với kết quả lúc train | Nhóm model | Giao cùng `predict` |

Khi có `pbind.predict`, endpoint `/predict` chỉ cần:

```python
from pbind import predict

pdb_text = call_esmfold_api(sequence)            # app: mục 2 (cache + kiểm tra + báo lỗi)
result = predict(sequence, binding_type="metal", pdb=pdb_text, profile="final",
                 weights_dir=r"D:\Protein_Ligand_Binding\model_weights",
                 final_model_dir=r"D:\Protein_Ligand_Binding\final_models\final_model")
```

Trước khi đưa lên ứng dụng thật, chạy `verify_installation.py` và phải ra PASS. Đó là cách chắc chắn môi trường trên máy chạy cho đúng kết quả như lúc train.
