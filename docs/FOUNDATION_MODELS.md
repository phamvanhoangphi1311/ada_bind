# Bốn model nền: cách dùng để ra đúng feature như lúc train

Mọi thiết lập dưới đây được lấy thẳng từ code đã tạo dữ liệu train:

| Bước | Code gốc |
|---|---|
| ESMFold | `legacy/notebooks/protein_folding_ESMFold_v1.ipynb` |
| ESM-2 | `ensure_esm_cache` trong backend `gear_ultimate/gear_structure.py` (port: `src/pbind/features/esm.py`) |
| ESM-IF | `multi_modal_binding/get_esm_if_embedding.py` của ESMBind, commit `5d81fb5b`, kèm bản vá thiết bị trong notebook 08 |
| ProtT5 | cell V3-3 của `notebooks/09_*` và `notebooks/10_*` |

Không đổi bất kỳ thiết lập nào. Model vẫn chạy và ra số khi sai thiết lập, nhưng số đó **sai mà không báo lỗi**.

## 0. File model

Thư mục `foundation_models/` (tạo bằng cell tải về, có `MANIFEST.json` ghi nguồn, revision và sha256):

| Thư mục | Nguồn (revision cố định) | File weight | sha256 |
|---|---|---|---|
| `esmfold_v1/` | `facebook/esmfold_v1` @ `75a3841e` | `pytorch_model.bin` (8.44 GB) | `2ee07356…e116` |
| `esm2_t33_650M_UR50D/` | `facebook/esm2_t33_650M_UR50D` @ `08e4846e` | `model.safetensors` (2.61 GB) | `a08adabb…4db0` |
| `prot_t5_xl_half_uniref50-enc/` | `Rostlab/prot_t5_xl_half_uniref50-enc` @ `94a6abc0` | `pytorch_model.bin` (2.42 GB) | `7f51ba88…1157` |
| `esm_if1/` | `dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt` | `.pt` (1,700,450,121 byte) | xem `MANIFEST.json` |

Cả bốn nguồn không thay đổi từ 2022–2023, nên đây đúng là các file đã dùng khi tạo PDB, tính feature và train.

## 1. Quy tắc chung

1. **Sequence chỉ gồm 20 amino acid chuẩn** (`ACDEFGHIKLMNPQRSTVWY`). Từ chối sequence có `X`, `U`, `B`, `Z`, `O`. Bốn protein bị loại khỏi Dev (C8BD48, P84801, Q00277, Q9NZV6) đều là vì lý do này.
2. **Độ dài ≤ 1000.** Dữ liệu train dài nhất 813. ESM-2 cắt ở 1000 residue, ESMFold ở 1024 token.
3. **Thứ tự chạy:** ESMFold → chuẩn hoá PDB về chain A → ESM-IF (cần PDB) → ESM-2 và ProtT5 (chỉ cần sequence) → feature viết tay → model.
4. **Chạy trên GPU,** đúng độ chính xác (fp16/fp32) ghi ở từng mục. Mỗi lần gọi model nền chỉ đưa vào **một protein**, trừ ESM-2 (xem mục 3).

## 2. ESMFold v1 → PDB

**Qua API (cách ứng dụng đang dùng):** gửi `POST https://api.esmatlas.com/foldSequence/v1/pdb/`, body là sequence dạng text thuần.
- **Kết quả:** PDB "ESMFOLD V1 PREDICTION", chain A, pLDDT thang 0–1, cùng định dạng với dữ liệu train.
- **Độ khớp:** so với PDB lúc train, RMSD Cα là 0.001 Å và 0.06 Å với protein có pLDDT 0.94 và 0.80; là 2.1 Å với protein có pLDDT 0.44. pLDDT lệch ≤ 0.03.
- **Giới hạn 400 aa** (dài hơn thì trả HTTP 413).

Muốn hỗ trợ chuỗi dài hơn thì tự host bằng đúng code dưới đây (mục `esmfold_v1` trong `MANIFEST.json`).

```python
import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

D = "foundation_models/esmfold_v1"
tokenizer = AutoTokenizer.from_pretrained(D)
model = EsmForProteinFolding.from_pretrained(D, low_cpu_mem_usage=True).cuda()
model.esm = model.esm.half()               # only the internal ESM-2 3B language model in fp16
model.trunk.set_chunk_size(64)
torch.backends.cuda.matmul.allow_tf32 = True

def fold(seq: str) -> str:
    ids = tokenizer([seq], return_tensors="pt", add_special_tokens=False, padding=True,
                    truncation=True, max_length=1024)["input_ids"].cuda()
    with torch.no_grad():
        out = model(ids)                   # num_recycles: default (do not pass)
    pos = atom14_to_atom37(out["positions"][-1], out)
    out = {k: v.to("cpu").numpy() for k, v in out.items()}
    prot = OFProtein(aatype=out["aatype"][0], atom_positions=pos.cpu().numpy()[0],
                     atom_mask=out["atom37_atom_exists"][0], residue_index=out["residue_index"][0] + 1,
                     b_factors=out["plddt"][0],
                     chain_index=out["chain_index"][0] if "chain_index" in out else None)
    return to_pdb(prot)                    # chain A, residues 1..L, B-factor = pLDDT on a 0-1 scale
```

- Cột B-factor chứa pLDDT trên **thang 0–1**. Giữ nguyên, không nhân 100: feature `CONF` lúc train được tính trên đúng file như vậy.
- **Không dùng `try/except/finally`** như vòng lặp gốc. Nếu fold lỗi, vòng lặp đó vẫn ghi PDB của protein trước dưới tên protein hiện tại. Fold lỗi thì phải báo lỗi.
- Cần GPU ≥ 16 GB. PDB gốc được fold trên Colab (GPU L4) vào 04/2025, bằng bản transformers mới nhất lúc đó (notebook chạy `pip install --upgrade transformers`).

### Chuẩn hoá PDB về chain A

Mọi feature cấu trúc, gồm cả ESM-IF, đều đọc PDB đã chuẩn hoá: chọn chain có sequence **trùng khít** sequence input, chỉ giữ residue chuẩn, đánh số lại 1..L, ghi lại bằng Biopython. Dùng `pbind.data.structures.normalize_exact_chain_A`; hàm này cho ra file giống từng byte với code gốc. Nếu không có chain nào trùng sequence thì báo lỗi, không đoán.

## 3. ESM-2 650M → `ESM` (L × 1280)

```python
import numpy as np, torch
from transformers import EsmModel, EsmTokenizer

D = "foundation_models/esm2_t33_650M_UR50D"
tok = EsmTokenizer.from_pretrained(D)
model = EsmModel.from_pretrained(D, output_hidden_states=True).cuda().eval()   # fp32, NOT half

def esm2(seqs: list[str]) -> list[np.ndarray]:          # original batch size: 8
    t = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=1002).to("cuda")
    with torch.inference_mode():
        h = torch.stack(model(**t).hidden_states[-2:], dim=0).mean(0)          # mean of the last two
    return [h[i, 1:len(s) + 1].float().cpu().numpy().astype(np.float16)        # drop BOS/EOS, store fp16
            for i, s in enumerate(seqs)]
```

- Phải load bằng **`transformers`** (`EsmModel`), không dùng `fair-esm`. "Hai hidden state cuối" là `hidden_states[-2]` = output của layer 32 và `hidden_states[-1]` = output layer 33 **sau** `emb_layer_norm_after`. transformers v4 (backend gốc) và v5 cho cùng định nghĩa; trên v5.20.2 đã kiểm tra bằng hook từng layer, lệch 0.0. Nếu tự gom output thô của layer 33 (chưa qua layer norm) thì sai lệch tới hàng nghìn đơn vị.
- Feature được lưu **fp16**; khi đưa vào model thì đổi sang float32.
- Gom batch có padding (lúc train là 8 protein/batch) chỉ làm lệch cỡ 1e-6 ở fp32, không ảnh hưởng gì.

## 4. ProtT5-XL half → `PROTT5` (L × 1024), chỉ nuclear + small

```python
import re, numpy as np, torch
from transformers import T5EncoderModel, T5Tokenizer   # needs sentencepiece + protobuf

D = "foundation_models/prot_t5_xl_half_uniref50-enc"
tok = T5Tokenizer.from_pretrained(D, do_lower_case=False)
t5 = T5EncoderModel.from_pretrained(D).cuda().eval().half()

def prott5(seq: str) -> np.ndarray:                     # one protein per call, no padding
    clean = re.sub(r"[UZOB]", "X", seq)
    ids = tok(" ".join(clean), add_special_tokens=True)["input_ids"]
    assert len(ids) == len(seq) + 1 and tok.unk_token_id not in ids[:-1]      # one token per residue + </s>
    ids = torch.tensor(ids)[None].cuda()
    with torch.no_grad():
        h = t5(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
    return h[0, :len(seq)].float().cpu().numpy().astype(np.float16)            # drop </s>, store fp16
```

## 5. ESM-IF1 → `ESMIF` (L × 512)

**Môi trường riêng** (bộ phiên bản đã chạy lúc tạo feature):

```bash
python 3.10
pip install "pip<25" setuptools wheel
pip install "numpy<2" scipy biopython tqdm ml-collections "biotite==0.41.2" "fair-esm==2.0.0"
pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.4.1"
pip install -f https://data.pyg.org/whl/torch-2.4.0+cu121.html "torch-scatter==2.1.2"
pip install "torch-geometric==2.6.1"
```

**Weight:** chép `esm_if1/esm_if1_gvp4_t16_142M_UR50.pt` vào `~/.cache/torch/hub/checkpoints/` (hoặc `$TORCH_HOME/hub/checkpoints/`). Khi đó `esm.pretrained.esm_if1_gvp4_t16_142M_UR50()` dùng file local, không tải lại. Dòng cảnh báo `Regression weights not found` là bình thường.

```python
import numpy as np, torch, esm, esm.inverse_folding
from esm.inverse_folding.util import CoordBatchConverter

model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.eval().to("cuda")                         # fp32

def get_encoder_output(model, alphabet, coords):       # = fair-esm's version + the device fix used for the features
    device = next(model.parameters()).device
    coords, confidence, _, _, padding_mask = CoordBatchConverter(alphabet)([(coords, None, None)], device=device)
    enc = model.encoder.forward(coords, padding_mask, confidence, return_all_hiddens=False)
    return enc["encoder_out"][0][1:-1, 0]               # drop BOS/EOS

def esmif(pdb_chain_a: str, seq: str) -> np.ndarray:
    structure = esm.inverse_folding.util.load_structure(pdb_chain_a, "A")
    coords, pdb_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
    assert pdb_seq == seq, "PDB chain A must match the sequence exactly"
    with torch.no_grad():
        return get_encoder_output(model, alphabet, coords).cpu().numpy()      # float32 (L, 512)
```

## 6. Kiểu dữ liệu khi đưa vào model đã train

Phải giữ đúng như lúc train:

| Feature | Metal (MLP) | Nuclear / small (hypergraph) |
|---|---|---|
| ESM-2 | fp16 → float32 | fp16 → float32 |
| ESM-IF | **float32 (không làm tròn)** | **làm tròn qua fp16** rồi đổi lại float32 |
| ProtT5 | — | fp16 → float32 |

Trong code inference của `pbind`, các bước trên nằm sẵn trong hàm. Bảng này dành cho ai tự viết lại.

## 7. Kiểm tra trước khi dùng thật

Một môi trường được coi là đúng khi tính lại bộ protein mẫu (golden set đi kèm `final_model/`, gồm sequence, PDB và feature lúc train) và đạt:

| Đại lượng | Ngưỡng đạt |
|---|---|
| PDB ESMFold | RMSD Cα ≤ 0.1 Å so với PDB lúc train; pLDDT lệch ≤ 0.01 |
| ESM-2 / ProtT5 / ESM-IF | cosine ≥ 0.999 mỗi residue; sai lệch tuyệt đối trung bình ≤ 1e-2 |
| Xác suất cuối (3 task) | lệch ≤ 1e-3 mỗi residue |

Golden set và script kiểm tra `scripts/verify_foundation_models.py` sẽ được tạo khi đóng gói `final_model/`, vì cần các file feature lúc train (zip ESM-IF / ProtT5 trên Drive). Trên GPU hoặc phiên bản thư viện khác, kết quả **không khớp từng bit** với lúc train nhưng vẫn nằm trong các ngưỡng trên. Ở bộ test, mức lệch đó thay đổi AUPRC khoảng 1e-4 (metal Test300: 0.38144 so với 0.381428).
