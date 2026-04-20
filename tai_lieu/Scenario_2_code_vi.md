# Scenario 2 — ANN Regression: Mô tả chi tiết code

## 1. Mục tiêu

Tái hiện **Scenario 2** trong paper (Section 2.2.2): tối ưu **trọng số của ANN feed-forward 1 hidden layer** để giải 9 bài toán hồi quy trên 3 dataset UCI × 3 kiến trúc hidden size.

**Mục tiêu tối ưu**: minimize MSE(ANN(θ; X), y) với dataset D = {(x⁽ⁱ⁾, y⁽ⁱ⁾)}.

Đây là **neuroevolution**: nghiệm `θ ∈ ℝᵖ` là vector weights của ANN, không phải điểm trong không gian toán học thông thường.

---

## 2. Cấu trúc file

```
Scenario_2/
├── Scenario_2_run.py        # Code chính
├── Scenario_2_local.ipynb   # Notebook phân tích
├── data/
│   ├── raw/                 # Dataset gốc tải từ UCI (.xls, .csv)
│   └── preprocessed/        # *.npz đã standardize + rescale y → [-1,1]
├── results/
│   └── Scenario_2_Novel_Final.csv
├── Figures/Scenario_2/      # 4 PNG
├── XLSX Files/Scenario_2/   # 1 XLSX
└── LaTeX/Scenario_2/        # 1 TXT
```

---

## 3. 9 Problems = 3 Datasets × 3 Architectures

### 3.1. Datasets (paper-exact)

| Dataset | m (features) | \|D\| | URL |
|---------|---|---|---|
| **Concrete** | 8 | 825 | `UCI /concrete/compressive/Concrete_Data.xls` |
| **Energy** | 8 | 615 | `UCI /242/energy+efficiency.zip` |
| **Wine** | 11 | 3919 | `UCI /wine-quality/winequality-white.csv` |

**Ghi chú Energy**: Dataset có 2 target (Y1=Heating, Y2=Cooling). Code chọn **Y1 (Heating Load)** để giữ single-output regression như paper.

**Ghi chú Wine**: Code drop duplicates rồi sample n=3919 theo paper.

### 3.2. Kiến trúc ANN

Feed-forward 1 hidden layer, **tanh activation** mọi neuron.

- Input: m neurons
- Hidden: `ρ_h × m` neurons, `ρ_h ∈ {1, 2, 3}`
- Output: 1 neuron

**Số tham số**: `p = (m × h) + h + (h × 1) + 1`, với h = ρ_h × m.

| Dataset | ρ_h=1 | ρ_h=2 | ρ_h=3 |
|---------|-------|-------|-------|
| Concrete (m=8) | p=81 | p=161 | p=241 |
| Energy (m=8) | p=81 | p=161 | p=241 |
| Wine (m=11) | p=144 | p=287 | p=430 |

Problem name: `ea.p.r.{Dataset}-{ρ_h}`, ví dụ `ea.p.r.Wine-3`.

### 3.3. Tiền xử lý

```python
X_std = (X_raw - X_raw.mean(0)) / X_raw.std(0)   # standardize features
y_01  = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
y_scaled = y_01 * 2 - 1                           # rescale y → [-1, 1]
```

Lý do rescale y: ANN output cuối dùng tanh nên chỉ ra được trong [-1, 1], vậy target cũng phải về [-1, 1].

Kết quả cached vào `data/preprocessed/{Concrete,Energy,Wine}.npz`.

---

## 4. Giải thích các hàm chính

### 4.1. Các hàm loader

```python
_load_concrete(data_dir)  → (X, y)
_load_energy(data_dir)    → (X, y)
_load_wine(data_dir)      → (X, y)
```

Mỗi loader:
1. Download từ UCI nếu file chưa có.
2. Parse `.xls`/`.csv`.
3. Trả về `(X_raw, y_raw)` chưa tiền xử lý.

### 4.2. `prepare_datasets(data_dir)`

1. Loop qua 3 datasets.
2. Nếu có `.npz` cache → đọc luôn.
3. Nếu không → load raw, standardize X, rescale y, cache `.npz`.
4. Return `dict {"Concrete": {"X": ..., "y": ...}, ...}`.

### 4.3. `_theta_dim(m, rho_h)` và `_unpack_theta`

Chuyển `θ ∈ ℝᵖ` flat vector thành `(W1, b1, W2, b2)`:

```python
h = rho_h * m
W1 shape = (m, h)
b1 shape = (h,)
W2 shape = (h, 1)
b2 scalar
```

### 4.4. `_predict(theta, X, rho_h)`

Forward pass:
```python
hidden = tanh(X @ W1 + b1)    # (n, h)
output = tanh(hidden @ W2 + b2)  # (n, 1)
```

### 4.5. `make_fitness_fn(X, y, rho_h)`

Closure trả về `fitness(theta) = MSE(predict(theta, X), y)`.

### 4.6. `_init_worker(preprocessed_dir)` + `_worker(args)`

**Tối ưu memory**: Mỗi process chỉ load dataset 1 lần (qua `_GLOBAL_DATA`), tránh pickle X/y to mỗi task.

Worker:
1. Lấy X, y từ global cache.
2. Build fitness với `rho_h` spec.
3. Gọi đúng solver (cùng 9 EAs như Scenario 1).
4. Trả records với metadata.

### 4.7. Solvers

**Y hệt** Scenario 1 (`run_cma_es`, `run_de`, `run_pso`, `run_es`, `run_ga`). Code gần như copy-paste — không có khác biệt về thuật toán, chỉ khác ở `fitness_fn`.

---

## 5. CLI Usage

```bash
# Full paper spec
python Scenario_2_run.py --n_rep 30 --n_evals 10000 --cores 4

# Smoke test
python Scenario_2_run.py --quick

# Chỉ 1 problem
python Scenario_2_run.py --problems ea.p.r.Concrete-1 --n_rep 5
```

**Args:**
- `--n_evals` (default 10000), `--n_rep` (default 30), `--cores` (default cpu_count-1).
- `--problems`: list subset problems.
- `--quick`: 2 reps, 800 evals, 1 core.

---

## 6. Output format (CSV)

**Y hệt Scenario 1** về schema. Khác biệt:
- `problem`: `ea.p.r.{Dataset}-{rho_h}` thay vì `ea.p.s.*`.
- `objective`: luôn `minimize`.
- `genotype_size`: p theo công thức mục 3.2.

---

## 7. Pipeline phân tích

Notebook `Scenario_2_local.ipynb` có cấu trúc giống S1:
1. Convergence plot
2. Effectiveness boxplot
3. NoVS (Wilcoxon α=0.01)
4. NER boxplot
5. EtTQ
6. Mean NER vs p

---

## 8. Mapping sang Paper

| Paper | Code |
|-------|------|
| Section 2.2.2 | `_load_*`, `prepare_datasets`, `_theta_dim`, `_predict` |
| 3 datasets (Concrete/Energy/Wine) | `UCI_*_URL`, `_load_*` |
| ANN architecture (tanh, ρ_h ∈ {1,2,3}) | `_unpack_theta`, `_predict` |
| Standardize X + rescale y ∈ [-1,1] | `_standardize_x`, `_rescale_y_to_minus1_1` |
| 9 problems | `build_problem_specs` |
| Figure 5 (convergence) | `_Convergence.png` |
| Figure 6 (NER + EtTQ) | `_NER.png`, `_Efficiency_percentile-0.75.png`, `_meanNER-vs-p.png` |
| Table 1 — Regression column | `_NOVS.xlsx` |

---

## 9. Kết quả thực nghiệm (30 seeds full run)

**NoVS**:

| EA | NoVS |
|----|------|
| **DE** | **9/9** |
| GA-0.02 | 1 |
| ES-0.02 | 1 |
| CMA-ES | 0 |
| Khác | 0 |

Khớp paper (Table 1, Regression): DE=7, GA-0.02=4, ES-0.02=3, CMA-ES=0.

**Kết luận quan trọng**: CMA-ES vốn thắng áp đảo S1 (synthetic) nhưng **thua hoàn toàn** ở S2 (regression) → hiệu năng trên benchmark tổng hợp không dự đoán được hiệu năng trên neuroevolution.

---

## 10. Ghi chú thực tế

- **Memory efficiency**: `_init_worker` cache dataset per-process thay vì re-pickle mỗi task → quan trọng vì Wine có |D|=3919 × 11 features, pickle nhiều lần sẽ chậm.
- **Wine dataset**: ~250s/run theo paper trên cluster Xeon. Local (4 cores) khoảng ~5–10 phút/run.
- **Reproducibility**: Wine sample với `seed=42` trong `_load_wine`. Nếu đổi seed sẽ có dataset khác → kết quả khác.
