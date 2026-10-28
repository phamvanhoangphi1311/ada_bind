# Gói `pbind` cho ứng dụng: cài đặt, kiểm tra, sử dụng

Gói này chứa phần mà checkpoint không có:
- **code tính feature** từ PDB (kể cả patch/pocket);
- **class kiến trúc model** (MLP, hypergraph);
- **cách gộp** các checkpoint thành xác suất cuối.

Tất cả được đóng thành một hàm. Trên máy của nhóm model, hàm này tính lại **đúng xác suất lúc train**: 16 protein test × 3 loại × 2 profile, sai lệch trung bình 2–4 × 10⁻⁵, **không residue nào đổi kết quả binding**.

## 1. Trong gói có gì

| File / thư mục | Dùng để |
|---|---|
| `pbind_src/` | package Python `pbind` (cài bằng pip) |
| `requirements-app.txt` | thư viện cần, kèm phiên bản đã kiểm chứng |
| `verify_installation.py` | kiểm tra cài đặt, báo PASS/FAIL |
| `golden_set/` | 12 protein mẫu: sequence, PDB, embedding lúc train, xác suất tham chiếu |
| `esmif_embed.py` | tính ESM-IF đúng như lúc train, chạy trong môi trường ESM-IF riêng |
| `manifest.json` | bản `final_model/manifest.json` đi cùng golden set này |
| `docs/APP_INTEGRATION.md`, `docs/FOUNDATION_MODELS.md` | kiến trúc, hợp đồng API, thiết lập từng model nền |

## 2. Cài đặt (môi trường chính của backend)

Python 3.10–3.12. Máy chỉ có CPU vẫn chạy được.

```bash
python -m venv .venv
.venv\Scripts\activate                  # Linux/WSL: source .venv/bin/activate
pip install -r requirements-app.txt     # có GPU NVIDIA: cài torch bản CUDA từ pytorch.org trước
pip install ./pbind_src
```

Nếu `D:\Protein_Ligand_Binding\final_models\final_model\manifest.json` khác file `manifest.json` trong gói, bước [1] của kiểm tra sẽ báo. Khi đó chép đè bằng file trong gói. Không sửa file này bằng tay.

## 3. Kiểm tra (bắt buộc trước khi dùng thật)

**Bước 1. Tạo embedding của golden set bằng chính code của app.**
- **ESM-IF:** chạy trong môi trường ESM-IF:
  ```bash
  python esmif_embed.py --golden-dir golden_set --out-dir my_embeddings ^
         --weights D:\Protein_Ligand_Binding\model_weights\esm_if1\esm_if1_gvp4_t16_142M_UR50.pt
  ```
- **ESM, ProtT5:** nếu app tự tính thì lưu thêm `my_embeddings\<prot_id>.ESM.npy` và `<prot_id>.PROTT5.npy` (shape `(L, 1280)` và `(L, 1024)`). Nếu để `pbind` tính thì bỏ qua.

**Bước 2. Chạy kiểm tra** (môi trường chính):

```bash
python verify_installation.py ^
  --final-model-dir D:\Protein_Ligand_Binding\final_models\final_model ^
  --weights-dir D:\Protein_Ligand_Binding\model_weights ^
  --golden-dir golden_set --embeddings-dir my_embeddings --device cpu
```

| Bước | Kiểm tra gì | Ngưỡng |
|---|---|---|
| [1] files | checksum 250 file final model; sha256 của model nền | khớp tuyệt đối |
| [2] pipeline | `pbind` + checkpoint trên input golden | lệch ≤ 1e-4 (đo được: 1e-7 trên CPU) |
| [3] embeddings | ESM / ProtT5 / ESM-IF máy này tạo ra so với lúc train | cosine ≥ 0.999 mỗi residue |
| [4] end-to-end | xác suất từ embedding của máy này so với lúc train | lệch ≤ 0.03 và gần như không residue nào đổi kết quả |

Phải ra `ALL PASS`. Có dòng `FAIL` thì gửi nguyên output cho nhóm model.

## 4. Dùng trong backend

Load **một lần** khi khởi động server. Mỗi request là một lần gọi `predict`:

```python
from pbind import Predictor, InputError

P = Predictor(r"D:\Protein_Ligand_Binding\final_models\final_model",
              r"D:\Protein_Ligand_Binding\model_weights",
              profile="final", device="cpu")            # "cuda" nếu có GPU

def handle(sequence: str, binding_type: str) -> dict:   # binding_type: "metal" | "nuclear" | "small"
    pdb_text = call_esmfold_api(sequence)              # ESM Atlas: xem docs/APP_INTEGRATION.md mục 2
    esmif = compute_esmif(pdb_text, sequence)          # môi trường ESM-IF: esmif_embed.embed(...), float32 (L, 512)
    try:
        return P.predict(sequence, binding_type, pdb=pdb_text, embeddings={"ESMIF": esmif})
    except InputError as e:                            # sequence / PDB / embedding không hợp lệ -> HTTP 400
        return {"error": str(e)}
```

- **`embeddings`:**
  - **ESMIF** bắt buộc, vì nó cần môi trường riêng.
  - **ESM, PROTT5** có thể truyền vào, nếu là code app đã qua kiểm tra [3]; không truyền thì `pbind` tự tính.
  - Truyền ESM-IF ở **float32 đúng như ESM-IF trả ra**, không tự làm tròn. Việc làm tròn fp16 cho nuclear/small, `pbind` tự làm, đúng như lúc train.
- **Kết quả:** dict theo đúng format ở mục 5 của `APP_INTEGRATION.md`, thêm `warnings`: chuỗi dài hơn mọi protein train (> 813 aa), hoặc pLDDT trung bình thấp (< 0.7).
- **Ngưỡng, trọng số, checkpoint** đều đọc từ `manifest.json`. Mỗi checkpoint được kiểm tra checksum trước khi load; file sai thì báo lỗi, không chạy.
- **`InputError`** cho các lỗi:
  - ký tự ngoài 20 amino acid chuẩn;
  - độ dài ngoài khoảng 30–1000;
  - PDB không có chain trùng khít sequence;
  - embedding sai kích thước.
- **Thời gian:** feature + checkpoint mất khoảng 0.2–0.3 giây/protein. Phần chậm là ESMFold API (7–25 giây) và, trên CPU, ESM-2 / ProtT5.

## 5. Không được làm

- Không đổi tên, không di chuyển file trong `final_model\`; không sửa `manifest.json`.
- Không dùng PDB của protein khác, hoặc PDB cũ, khi ESMFold API lỗi. Phải báo lỗi.
- Không tự viết lại feature hay model. Mọi tính toán phải đi qua `pbind`, như vậy kết quả mới đúng như đã kiểm chứng.
