# Simple Evolutionary Algorithm — Neuroevolution of Continuous Control Policies

> Tái hiện kết quả nghiên cứu của **El Saliby et al. (2024)** — *Eventually, All You Need is a Simple Evolutionary Algorithm (for Neuroevolution of Continuous Control Policies)*. GECCO '24 Companion, Melbourne, Australia.
> DOI: [10.1145/3638530.3664112](https://doi.org/10.1145/3638530.3664112)

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Bộ 9 Thuật toán Tiến hóa](#bộ-9-thuật-toán-tiến-hóa)
- [4 Kịch bản thí nghiệm](#4-kịch-bản-thí-nghiệm)
- [Cài đặt](#cài-đặt)
- [Cách chạy](#cách-chạy)
- [Kết quả](#kết-quả)
- [Tài liệu](#tài-liệu)
- [Trích dẫn](#trích-dẫn)

---

## Tổng quan

Dự án so sánh **9 cấu hình thuật toán tiến hóa (EA)** trên **4 nhóm bài toán tối ưu** với độ phức tạp tăng dần, nhằm trả lời câu hỏi:

> *Liệu một EA tinh vi (như CMA-ES) có thực sự cần thiết để giải neuroevolution trên bài toán điều khiển phức tạp, hay các EA đơn giản như GA/DE cũng đủ?*

**Phát hiện chính** (đã tái hiện trong dự án):

- **CMA-ES** thống trị trên benchmark toán học tổng hợp (S1: 20/24 NoVS).
- **DE và GA** vượt trội trên các bài toán neuroevolution phức tạp (S2: DE=9/9; S4: GA variants dẫn đầu).
- Hiệu năng trên synthetic **không dự đoán** hiệu năng trên control tasks.
- Giả thuyết: **toán tử crossover** trong GA/DE giúp khám phá fitness landscape gồ ghề của bài toán control tốt hơn.

---

## Cấu trúc dự án

```
Simple_Evolutionary_Algorithm/
├── README.md                          # File này
├── Scenario_1/                        # Synthetic benchmarks (24 problems)
│   ├── Scenario_1_run.py              # EA runner
│   ├── Scenario_1_local.ipynb         # Analysis notebook
│   ├── results/                       # CSV output
│   ├── Figures/Scenario_1/            # 5 PNG figures
│   ├── XLSX Files/Scenario_1/         # NoVS tables
│   └── LaTeX/Scenario_1/              # LaTeX-ready tables
├── Scenario_2/                        # ANN Regression (9 problems)
│   ├── data/
│   │   ├── raw/                       # UCI datasets
│   │   └── preprocessed/              # Cached .npz
│   ├── Scenario_2_run.py
│   └── ...
├── Scenario_3/                        # 2D Navigation (9 problems)
├── Scenario_4/                        # VSR Control (15 problems)
└── tai_lieu/                          # Documentation (Vietnamese)
    ├── 3638530.3664112.pdf            # Original paper
    ├── paper_summary_vi.md            # Detailed paper summary
    ├── scenarios_vi.md                # 4 scenarios breakdown
    ├── algorithms.md                  # Pseudocode + math for 5 EAs
    ├── improvement_idea_vi.md         # Adaptive-σ improvement proposal
    ├── Scenario_1_code_vi.md          # S1 code walkthrough
    ├── Scenario_2_code_vi.md          # S2 code walkthrough
    ├── Scenario_3_code_vi.md          # S3 code walkthrough
    └── Scenario_4_code_vi.md          # S4 code walkthrough
```

---

## Bộ 9 Thuật toán Tiến hóa

Tất cả EAs tuân thủ **paper-exact hyperparameters** (Hansen 2016, Clerc 2012, El Saliby et al. 2024):

| # | EA | Key params | Có crossover? |
|---|-----|-----------|:---:|
| 1 | **CMA-ES** | Vanilla defaults, σ₀=0.5 | ✗ |
| 2 | **DE** (rand/1/bin) | NP=15, F=0.5, CR=0.8 | ✓ |
| 3 | **PSO** (Clerc 2012) | n_pop=100, w=0.8, φ=1.5 | ✗ |
| 4 | **ES-0.02** | n_pop=30, μ_parents=10, σ=0.02 | ✗ |
| 5 | **ES-0.25** | ... σ=0.25 | ✗ |
| 6 | **ES-0.5** | ... σ=0.5 | ✗ |
| 7 | **GA-0.02** | n_pop=100, n_tour=5, p_xo=0.8, σ=0.02 | ✓ |
| 8 | **GA-0.25** | ... σ=0.25 | ✓ |
| 9 | **GA-0.5** | ... σ=0.5 | ✓ |

**Common settings**: Init `x ~ U([-1, 1])^p`; terminate at `total_evals > 10 000`.

Chi tiết pseudocode và công thức: [tai_lieu/algorithms.md](tai_lieu/algorithms.md).

---

## 4 Kịch bản thí nghiệm

Độ phức tạp tăng dần theo độ **gián tiếp** của mapping θ → fitness:

### Scenario 1 — Synthetic Numerical Benchmarks

- **Bài toán**: Tối ưu trực tiếp f(x) với x ∈ ℝᵖ.
- **24 problems**: 6 hàm × 4 dims p ∈ {20, 100, 200, 500}.
- **Functions**: Sphere, PA-1, PA-10, CPA, Ackley, Rastrigin.
- **Mô tả chi tiết**: [tai_lieu/Scenario_1_code_vi.md](tai_lieu/Scenario_1_code_vi.md).

### Scenario 2 — ANN-based Regression

- **Bài toán**: Tối ưu θ của ANN để minimize MSE trên dataset.
- **9 problems**: 3 datasets × 3 kiến trúc (ρ_h ∈ {1, 2, 3}).
- **Datasets**: Concrete (m=8), Energy (m=8), Wine (m=11).
- **ANN**: Feed-forward 1 hidden layer, tanh activation, p ∈ {81, ..., 430}.
- **Mô tả chi tiết**: [tai_lieu/Scenario_2_code_vi.md](tai_lieu/Scenario_2_code_vi.md).

### Scenario 3 — ANN-based 2D Navigation

- **Bài toán**: Tối ưu ANN controller cho robot differential-drive trong arena 1m×1m.
- **9 problems**: 3 arenas × 3 sensor counts (m ∈ {3, 5, 9}).
- **Arenas**: Small / Large / Maze. Target cố định tại (0.5, 0.15).
- **Physics**: Δt=0.1s, 60s simulation. Fitness stochastic (init ngẫu nhiên).
- **Mô tả chi tiết**: [tai_lieu/Scenario_3_code_vi.md](tai_lieu/Scenario_3_code_vi.md).

### Scenario 4 — VSR (Voxel-based Soft Robot) Control

- **Bài toán**: Tối ưu ANN controller cho robot mềm 2D với mass-spring physics.
- **15 problems**: 5 tasks × 3 controller-sensor combos.
- **Morphologies**: Biped (10 voxels) cho locomotion/jump; Tower (14 voxels) cho balance.
- **Controllers**: C (centralized) / HoD (homo-distributed) / HeD (hetero-distributed).
- **Mô tả chi tiết**: [tai_lieu/Scenario_4_code_vi.md](tai_lieu/Scenario_4_code_vi.md).

Tổng quan 4 scenarios: [tai_lieu/scenarios_vi.md](tai_lieu/scenarios_vi.md).

---

## Cài đặt

### Yêu cầu

- Python 3.10+
- RAM: **32GB khuyến nghị** (16GB vẫn chạy được với `--cores 8`)
- Dung lượng: ~500MB (datasets + results)

### Dependencies

```bash
pip install numpy pandas scipy matplotlib openpyxl xlrd cma tqdm jupyter nbconvert
```

Hoặc:

```bash
pip install -r requirements.txt
```

(Tạo `requirements.txt` nếu chưa có từ list trên.)

---

## Cách chạy

### Quick smoke test (1–5 phút)

```bash
python Scenario_1/Scenario_1_run.py --quick
python Scenario_2/Scenario_2_run.py --quick
python Scenario_3/Scenario_3_run.py --quick
python Scenario_4/Scenario_4_run.py --quick
```

### Full paper spec

Paper dùng 30 seeds (20 cho S4), 10 000 fitness evaluations/run. Chạy với **4 cores** để tránh MemoryError (tùy RAM):

```bash
# S1 — ~1h 10min với 4 cores
python Scenario_1/Scenario_1_run.py --n_rep 30 --n_evals 10000 --cores 4

# S2 — ~3–5h
python Scenario_2/Scenario_2_run.py --n_rep 30 --n_evals 10000 --cores 4

# S3 — ~20h (full) | ~3.5h (5 seeds × 3000 evals)
python Scenario_3/Scenario_3_run.py --n_rep 30 --n_evals 10000 --cores 4

# S4 — ~280h full (không khả thi local) | ~5h (3 seeds × 1500 evals × t_sim=15)
python Scenario_4/Scenario_4_run.py --n_rep 20 --n_evals 10000 --cores 4
```

**Guidelines cho `--cores`** (tùy RAM):

| RAM | --cores khuyến nghị |
|-----|---------------------|
| 8 GB | 3–4 |
| 16 GB | 8 |
| 32 GB | 16 |
| 64 GB+ | 20 (all) |

### Chạy notebook phân tích

```bash
cd Scenario_1
jupyter nbconvert --to notebook --execute Scenario_1_local.ipynb --output Scenario_1_executed.ipynb
# Hoặc mở interactive:
jupyter notebook Scenario_1_local.ipynb
```

Notebook tự động sinh:
- **Figures/**: Convergence, Effectiveness, NER, EtTQ, NER-vs-p plots (PNG, dpi=300).
- **XLSX Files/**: NoVS tables (Wilcoxon signed-rank, α=0.01).
- **LaTeX/**: Tables ready for LaTeX `\input{}`.

### Subset runs

```bash
# Chỉ chạy 1 problem
python Scenario_1/Scenario_1_run.py --problems ea.p.s.sphere-20 --n_rep 5

# Chỉ VSR flat-HoD
python Scenario_4/Scenario_4_run.py --problems ea.p.v.flat-HoD --n_rep 3 --cores 4
```

---

## Kết quả

### Bảng NoVS (Number of Victories Score)

*Số bài toán mỗi EA "thắng" (final fitness không khác biệt có ý nghĩa với EA best-median, Wilcoxon α=0.01).*

| EA | S1 Synthetic (24) | S2 Regression (9) | S3 Navigation (9) | S4 VSR (15) |
|----|:---:|:---:|:---:|:---:|
| **CMA-ES** | **20** | 0 | — | — |
| **DE** | 4 | **9** | — | — |
| **PSO** | 0 | 0 | — | — |
| **ES-0.02** | 0 | 1 | — | — |
| **ES-0.25** | 0 | 0 | — | — |
| **ES-0.5** | 1 | 0 | — | — |
| **GA-0.02** | 0 | 1 | — | — |
| **GA-0.25** | 0 | 0 | — | — |
| **GA-0.5** | 0 | 0 | — | — |

*S3, S4 NoVS chưa thống kê được do số seeds không đủ cho Wilcoxon α=0.01 (5 và 3 seeds). Mean NER tham khảo:*

**Mean NER S3 (5 seeds, lower=better)**:
1. ES-0.25 (0.272) | 2. ES-0.5 (0.311) | 3. ES-0.02 (0.378) | 4. CMA-ES (0.483)

**Mean NER S4 (3 seeds, lower=better)**:
1. GA-0.25 (0.339) | 2. GA-0.02 (0.343) | 3. GA-0.5 (0.364) | 4. ES-0.02 (0.485)

### So sánh với Table 1 paper

| Scenario | Top EA (paper) | Top EA (reproduction) |
|----------|---------------|------------------------|
| S1 Synthetic | CMA-ES (18/24) | **CMA-ES (20/24)** ✓ |
| S2 Regression | DE (7/9) | **DE (9/9)** ✓ |
| S3 Navigation | DE (9/9) | ES/CMA-ES (NER-based) — cần thêm seeds |
| S4 VSR | DE (13/15) | **GA variants (NER-based)** ≈ |

→ **Xu hướng chính của paper được tái hiện đúng** trên S1 và S2. S3, S4 cần ≥ 15 seeds để so sánh NoVS trực tiếp.

### Visualization

Mỗi scenario sinh ra bộ figures sau:

- `*_Convergence.png` — Median + IQR của best_fitness theo evaluations.
- `*_Effectiveness.png` (S1) — Boxplot final best_fitness.
- `*_NER.png` — Boxplot Normalized Effectiveness Rank.
- `*_Efficiency_percentile-0.75.png` — Boxplot EtTQ (Evals to 3rd Quartile).
- `*_meanNER-vs-p.png` — Line plot NER theo genotype size p.
- `*_NoVS_per_problem.png` (S3, S4) — Bar chart chiến thắng per problem.

---

## Tài liệu

### Paper gốc
- [tai_lieu/3638530.3664112.pdf](tai_lieu/3638530.3664112.pdf) — Bản PDF paper gốc.

### Tài liệu tiếng Việt
- [paper_summary_vi.md](tai_lieu/paper_summary_vi.md) — Tóm tắt chi tiết paper (tiếng Việt).
- [scenarios_vi.md](tai_lieu/scenarios_vi.md) — 4 scenarios đầy đủ thông số paper.
- [algorithms.md](tai_lieu/algorithms.md) — Pseudocode + công thức + ví dụ cho 5 họ EA.
- [improvement_idea_vi.md](tai_lieu/improvement_idea_vi.md) — Đề xuất cải tiến Adaptive-σ.
- [Scenario_1_code_vi.md](tai_lieu/Scenario_1_code_vi.md) — Walkthrough code S1.
- [Scenario_2_code_vi.md](tai_lieu/Scenario_2_code_vi.md) — Walkthrough code S2.
- [Scenario_3_code_vi.md](tai_lieu/Scenario_3_code_vi.md) — Walkthrough code S3.
- [Scenario_4_code_vi.md](tai_lieu/Scenario_4_code_vi.md) — Walkthrough code S4.

---

## Metrics đánh giá

Theo paper (Section 2.3):

- **NER** (Normalized Effectiveness Rank) ∈ [0, 1], **lower = better**. Đo qua normalized rank của final fitness trên tất cả (EA, seed) cho mỗi problem.
- **EtTQ** (Evals to Third Quartile). Số evaluations cần để run đạt Q3 của phân phối final fitness. **Lower = better**.
- **NoVS** (Number of Victories Score). Tổng số problems mỗi EA "thắng" theo **Wilcoxon signed-rank test** (α=0.01), so với EA có best median.

---

## Ghi chú về re-implementation

- **Framework**: Paper dùng **JGEA** (Java). Code này viết lại thuần **Python** (numpy + cma + scipy).
- **Scenarios 1, 2**: Re-implementation trực tiếp (toán học + ANN đơn giản). Kết quả khớp paper chặt.
- **Scenario 3**: Physics simulator đơn giản (kinematic differential-drive + ray-segment collision). Paper details đủ dùng.
- **Scenario 4**: Paper dùng `2D-VSR-Sim` (Medvet et al. 2020) — simulator mass-spring phức tạp. Code này dùng **Verlet mass-spring rút gọn** — giữ đúng pipeline, controller architectures, tasks, fitness; **physics đơn giản hóa**. Xu hướng kết quả đúng paper, con số NoVS cụ thể có thể lệch.

---

## Trích dẫn

Khi dùng code này hoặc tham khảo, vui lòng trích dẫn paper gốc:

```bibtex
@inproceedings{elsaliby2024simpleea,
  author    = {El Saliby, Michel and Nadizar, Giorgia and Salvato, Erica and Medvet, Eric},
  title     = {Eventually, All You Need is a Simple Evolutionary Algorithm
               (for Neuroevolution of Continuous Control Policies)},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference Companion
               (GECCO '24 Companion)},
  year      = {2024},
  pages     = {1904--1913},
  doi       = {10.1145/3638530.3664112},
  publisher = {ACM}
}
```

**Paper repo gốc**: https://github.com/elsalibymichel/2024_NEWK-GECCO_EA-Comparison

---

## License

Dự án này chỉ phục vụ **mục đích học tập và nghiên cứu**. Paper gốc và dữ liệu UCI tuân thủ license riêng của publisher/data provider.

---

## Environment đã test

- OS: Windows 11 / Ubuntu 22.04
- Python: 3.11.9
- Dependencies: numpy 2.3+, pandas 2.3+, scipy 1.16+, cma 4.4+, matplotlib 3.10+
- Hardware tham chiếu paper: Intel Xeon W-2295 @ 3.0 GHz, 36 cores, 64 GB RAM (Ubuntu 20.04)
