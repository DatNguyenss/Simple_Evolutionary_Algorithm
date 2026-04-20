# Scenario 1 — Synthetic Benchmarks: Mô tả chi tiết code

## 1. Mục tiêu

Tái hiện **Scenario 1** trong paper *El Saliby et al. 2024* (Section 2.2.1): so sánh 9 thuật toán tiến hóa (EA) trên **24 bài toán tối ưu số học tổng hợp** — 6 hàm benchmark × 4 kích thước không gian `p ∈ {20, 100, 200, 500}`.

**Mục tiêu tối ưu**: minimize f(x), x ∈ ℝᵖ.

---

## 2. Cấu trúc file

```
Scenario_1/
├── Scenario_1_run.py        # Code chính: chạy 9 EAs × 24 problems × 30 reps
├── Scenario_1_local.ipynb   # Notebook phân tích: NER, EtTQ, NoVS, figures
├── results/
│   └── Scenario_1_Novel_Final.csv   # Output CSV (372k dòng khi full)
├── Figures/Scenario_1/      # 5 PNG figures
├── XLSX Files/Scenario_1/   # 2 XLSX summary
└── LaTeX/Scenario_1/        # TXT files cho LaTeX
```

---

## 3. Bộ 9 EAs — paper-exact hyperparameters

| Key | Tên | Tham số paper |
|-----|-----|----------------|
| `CMA-ES`  | Vanilla CMA-ES | default Hansen (2016), σ₀=0.5 |
| `DE`      | DE/rand/1/bin  | NP=15, F=0.5, CR=0.8 |
| `PSO`     | Clerc 2012     | n_pop=100, w=0.8, φ_p=φ_g=1.5 |
| `ES-σ`    | Basic ES (μ,λ) | n_pop=30, n_parents=10, σ ∈ {0.02, 0.25, 0.5} |
| `GA-σ`    | Real-valued GA | n_pop=100, n_tour=5, p_xo=0.8, σ ∈ {0.02, 0.25, 0.5} |

**Khởi tạo chung**: `x ~ U([-1, 1])^p` cho mọi EA.
**Điều kiện dừng**: `total_evals > n_evals` (mặc định 10 000).

---

## 4. Giải thích các hàm chính

### 4.1. Benchmark functions (`sphere`, `point_aiming`, `circular_point_aiming`, `ackley`, `rastrigin`)

Tất cả **nhận input vector x ∈ ℝᵖ** và trả về fitness float.

```python
def sphere(x):         return sum(x_i²)
def point_aiming(x, target):    return ||x - target||
def circular_point_aiming(x, targets):  return min_i ||x - target_i||
def ackley(x):         công thức Ackley chuẩn
def rastrigin(x):      công thức Rastrigin chuẩn
```

**Lưu ý CPA**: 5 điểm target cố định trên vòng tròn (tâm **1**, bán kính 2) theo seed 42 — đảm bảo cùng target cho mọi experiment với cùng p.

### 4.2. Hàm `_record(iteration, total_evals, ...)`

Ghi 1 record theo schema giống JGEA:
- `iterations`, `evals`, `births`, `elapsed`
- `all_size`, `firsts_size`, `lasts_size`
- `geno_uni`, `sol_uni`, `fit_uni` (diversity metrics)
- `best_fitness`

Record được append **mỗi 5 iterations** (không phải mỗi iter) để giảm kích thước CSV.

### 4.3. Các hàm solver

Tất cả solver trả về **list of records**, **không** trả best solution (chỉ cần fitness để tính NER/NoVS).

#### `run_cma_es(fitness_fn, dim, seed, n_evals)`
- Dùng thư viện `cma.CMAEvolutionStrategy`.
- `x0 = rng.uniform(-1, 1, dim)` (paper init, không phải default của cma).
- `σ₀ = 0.5` (default).
- Loop: `ask()` → evaluate → `tell()`.

#### `run_de(fitness_fn, dim, seed, n_evals)`
- DE/rand/1 per-target mutation: `v = x_r1 + F(x_r2 - x_r3)`.
- Binomial crossover theo chiều: mask ~ rand(dim) < CR.
- Selection 1–1: `x_i ← u` nếu `f(u) ≤ f(x_i)`.

#### `run_pso(fitness_fn, dim, seed, n_evals)`
- Vector hóa cả swarm (100 particles).
- Cập nhật velocity: `v = w*v + φp*r1*(pbest - pos) + φg*r2*(gbest - pos)`.
- Update pbest/gbest sau mỗi iteration.

#### `run_es(fitness_fn, dim, seed, sigma, n_evals)`
- Basic (μ, λ) = (30, 30) với μ_parents = 10.
- **Elitism**: giữ best cá thể sang thế hệ sau.
- Sample n_pop-1 cá thể mới: `μ + N(0, σI)`.
- Lưu ý: σ trong code là **variance** (`std = sqrt(sigma)`) theo paper notation.

#### `run_ga(fitness_fn, dim, seed, sigma, n_evals)`
- **Tournament selection** kích thước n_tour=5.
- **Gaussian mutation**: `x' = x + N(0, σI)`.
- **Segment geometric crossover**: `x' = x₁ + α(x₂ - x₁) + N(0, σI)`, α ~ U(0,1) scalar.
- **Replacement**: merge (parents ∪ offspring), giữ top n_pop.

### 4.4. Hàm `_worker(args)`

Được gọi bởi multiprocessing pool. Mỗi worker:
1. Unpack args: `(prob_key, jname, fn_tag, dim, obj, solver_key, sigma, seed, n_evals)`.
2. Build fitness function `fn = _make_fitness(fn_tag, dim)`.
3. Gọi đúng solver theo `solver_key`.
4. Gắn metadata `(seed, problem, solver_sigma, objective, genotype_size)` vào mỗi record.

### 4.5. Hàm `build_problems(dims)`

Sinh ra 24 problems: 6 hàm × 4 dims. Ví dụ problem name: `ea.p.s.sphere-20`, `ea.p.s.pointAiming-100-10`.

### 4.6. `main()`

1. Parse CLI args.
2. Build task list: cartesian product `problems × solvers × seeds`.
3. Chạy song song qua `multiprocessing.Pool` (spawn context).
4. Gom tất cả records, lưu CSV.

---

## 5. CLI Usage

```bash
# Full run (paper spec): 30 seeds × 24 problems × 9 EAs = 6480 tasks
python Scenario_1_run.py --n_rep 30 --n_evals 10000 --cores 4

# Smoke test (1–2 phút)
python Scenario_1_run.py --quick

# Subset: chỉ Sphere-20 với 5 reps
python Scenario_1_run.py --problems ea.p.s.sphere-20 --n_rep 5
```

**Args:**
- `--n_evals` (int, default 10000): budget fitness evaluations/run.
- `--n_rep` (int, default 30): số seed/(EA, problem).
- `--cores` (int, default cpu_count-1): số worker song song.
- `--dims` (list int): override dims (default [20, 100, 200, 500]).
- `--problems` (list str): chỉ chạy subset problems.
- `--quick`: smoke test (2 reps, 500 evals, dim=[20], 1 problem).

---

## 6. Output format (CSV)

Separator `;`. Columns:

| Column | Ý nghĩa |
|--------|---------|
| `iterations` | iteration index |
| `evals` | Tổng fitness evaluations tới thời điểm record |
| `births` | Số cá thể đã sinh |
| `elapsed` | Giây từ lúc bắt đầu run |
| `all_size` | Kích thước quần thể |
| `firsts_size`, `lasts_size` | Size meta-data |
| `geno_uni`, `sol_uni`, `fit_uni` | Diversity (unique / pop_size) |
| `best_fitness` | Best fitness tới thời điểm record |
| `seed` | Seed EA (1-based) |
| `problem` | Problem name (e.g. `ea.p.s.sphere-20`) |
| `solver_sigma` | EA variant (`cmaEs`, `differentialEvolution`, `es-0.5`, ...) |
| `objective` | `minimize` |
| `best→fitness` | Giống `best_fitness` (alias JGEA) |
| `genotype_size` | p |

---

## 7. Pipeline phân tích (notebook `Scenario_1_local.ipynb`)

1. **Đọc CSV** → dataframe 1.8M rows.
2. **Parse solver/sigma** từ `solver_sigma`.
3. **Convergence plot**: median + Q1–Q3 shaded theo evals.
4. **Effectiveness boxplot**: boxplot final best_fitness per (problem, solver_sigma).
5. **NoVS (Wilcoxon signed-rank, α=0.01)**: so với best-median EA cho mỗi problem.
6. **NER boxplot**: normalized rank final fitness ∈ [0, 1], càng nhỏ càng tốt.
7. **EtTQ**: evals cần để đạt Q3 (75th percentile) của toàn bộ final fitnesses.
8. **Mean NER vs p**: line plot xu hướng theo genotype size.

Output lưu vào `Figures/`, `XLSX Files/`, `LaTeX/`.

---

## 8. Mapping sang Paper

| Paper | Code |
|-------|------|
| Section 2.1 — EAs | `run_cma_es`, `run_de`, `run_pso`, `run_es`, `run_ga` |
| Section 2.2.1 — 24 problems | `build_problems`, benchmark fns |
| Section 2.3.1 — NER | Cell 13 của notebook |
| Section 2.3.1 — NoVS | Cell 11 của notebook |
| Section 2.3.2 — EtTQ | Cell 15 của notebook |
| Figure 3 (convergence) | `_Convergence.png` |
| Figure 4 (NER + EtTQ + NER vs p) | `_NER.png`, `_Efficiency_percentile-0.75.png`, `_meanNER-vs-p.png` |
| Table 1 (NoVS) | `_NOVS.xlsx` |

---

## 9. Kết quả thực nghiệm (30 seeds, 24 problems)

**NoVS** (theo Wilcoxon α=0.01):

| EA | NoVS |
|----|------|
| CMA-ES | **20** |
| DE | 4 |
| ES-0.5 | 1 |
| Khác | 0 |

Trùng xu hướng paper (Table 1: CMA-ES=18/24, GA-0.02=6, ES-0.02=2).
