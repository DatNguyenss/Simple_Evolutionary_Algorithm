# Scenario 3 — ANN 2D Navigation: Mô tả chi tiết code

## 1. Mục tiêu

Tái hiện **Scenario 3** trong paper (Section 2.2.3): tối ưu **trọng số của ANN controller** cho robot 2-bánh differential-drive điều hướng trong arena 1m × 1m, tìm target cố định.

**Bài toán**: 9 problems = 3 arenas × 3 sensor counts.
**Fitness**: khoảng cách trung bình robot–target trong 60s mô phỏng (MINIMIZE, stochastic).

---

## 2. Cấu trúc file

```
Scenario_3/
├── Scenario_3_run.py        # Code chính: EA solvers + physics simulator + ANN
├── Scenario_3_local.ipynb   # Notebook phân tích
├── results/
│   └── Scenario_3_Novel_Final.csv
├── Figures/Scenario_3/      # 5 PNG
├── XLSX Files/Scenario_3/   # 1 XLSX
└── LaTeX/Scenario_3/        # 2 TXT
```

---

## 3. Môi trường mô phỏng (paper-exact)

### 3.1. Hằng số vật lý

```python
ARENA_W = ARENA_H = 1.0    # m
R_ROBOT  = 0.05            # m (robot radius)
V_MAX    = 0.10            # m/s (max wheel speed)
SENSOR_R = 1.0             # m (proximity range)
DT       = 0.1             # s (physics step)
T_SIM    = 60.0            # s (simulation duration)
TARGET   = (0.50, 0.15)    # fixed target position
```

### 3.2. Khởi tạo ngẫu nhiên (stochastic fitness)

```python
x0 ~ U([0.45, 0.55])
y0 ~ U([0.80, 0.85])
heading ~ U([0, 2π])
```

Mỗi fitness call dùng 1 RNG mới seed bằng `(EA_seed * 10_000_000 + eval_counter)` → đảm bảo:
- **Reproducibility**: cùng EA seed → cùng chuỗi initial conditions.
- **Stochasticity**: mỗi evaluation thấy 1 episode khác → mimicking paper's stochastic fitness.

### 3.3. 3 Arenas (Figure 1 paper)

```python
ARENAS = {
  "Small": [(0.40, 0.30, 0.60, 0.30)],                        # 1 barrier ngắn
  "Large": [(0.20, 0.30, 0.80, 0.30)],                        # 1 barrier dài
  "Maze":  [(0.20, 0.40, 0.60, 0.40), (0.40, 0.20, 0.80, 0.20)]  # 2 barriers mê cung
}
```

Mỗi barrier là segment `(x1, y1, x2, y2)`. 4 tường arena tự động thêm.

---

## 4. Kiến trúc ANN Controller

```python
n_in  = m + 2           # m proximity + dist_to_target + angle_to_target
n_hid = 3 * (m + 2)     # hidden = 3 × input (paper)
n_out = 2               # left wheel speed, right wheel speed
```

Activation: tanh mọi nơi. Output tanh ∈ [-1, 1] mapped trực tiếp sang wheel speeds ∈ [-V_MAX, V_MAX] (cho phép lùi / quay tại chỗ).

**Số tham số**:

| m | p |
|---|---|
| 3 | 122 |
| 5 | 212 |
| 9 | 464 |

Khớp paper.

---

## 5. Giải thích các hàm chính

### 5.1. Hàm hình học

#### `_seg_dist(px, py, ax, ay, bx, by)`
Khoảng cách nhỏ nhất từ điểm (px, py) đến segment AB.

#### `_ray_seg_dist(ox, oy, angle, ax, ay, bx, by, max_r)`
Khoảng cách dọc tia từ (ox, oy) theo hướng `angle` đến segment AB, capped bởi `max_r`.
Dùng công thức intersection ray-line thuần đại số.

#### `_proximity_reading(ox, oy, angle, barriers, max_r)`
Reading của 1 proximity sensor:
- Tính min distance từ (ox, oy) theo tia `angle` đến **mọi barrier + 4 walls**.
- Return `d / max_r` ∈ [0, 1]: **1 = không có gì trong range, 0 = chạm trực tiếp**.

#### `_min_dist_to_obstacles(px, py, barriers)`
Khoảng cách min từ điểm tới mọi barrier. Dùng để kiểm tra va chạm.

### 5.2. Hàm ANN Forward

#### `_theta_dim_nav(m)`
Tính số tham số theo công thức mục 4.

#### `_ann_forward(theta, inputs, n_in, n_hid, n_out)`
2-layer tanh:
```python
h = tanh(inputs @ W1 + b1)
out = tanh(h @ W2 + b2)
```

### 5.3. Hàm mô phỏng — `simulate_navigation(theta, m, arena_name, rng_sim, t_sim=60.0)`

**Đây là core logic**:

```
1. Setup: unpack weights, random init (x, y, heading)
2. Loop 600 steps (60s / 0.1s):
   a. Đọc m proximity sensors (góc phân bố đều trên frontal half-arc [-π/2, π/2])
   b. Tính distance + angle tới target (2 sensors phụ)
   c. Forward ANN → (v_left, v_right) ∈ [-V_MAX, V_MAX]
   d. Differential drive:
        v = (v_L + v_R) / 2
        ω = (v_R - v_L) / (2 * R_ROBOT)
        heading += ω * DT
        new_x = x + v·cos(heading)·DT
        new_y = y + v·sin(heading)·DT
   e. Collision handling:
        - Clamp (new_x, new_y) vào arena (trừ đi R_ROBOT)
        - Nếu _min_dist_to_obstacles < R_ROBOT → stay put
   f. total_dist += euclidean(x, y, TARGET)
3. return total_dist / steps   # mean distance (minimize)
```

**Chi tiết angle-to-target sensor**:
```python
a_rel = atan2(dy, dx) - heading
a_rel = wrap_to_pi(a_rel)       # ∈ [-π, π]
sensor = a_rel / π              # ∈ [-1, 1]
```
Paper cho phép kiểu wrap này (full angle, không chỉ sin) để tránh ambiguity front-vs-back.

### 5.4. `build_problem_specs()`

Sinh 9 problems: `ea.p.n.{Small,Large,Maze}-{3,5,9}`.

### 5.5. `_worker(args)`

**Điểm đặc biệt của S3**: fitness là **stochastic**, mỗi eval thấy init khác nhau. Cách implement:

```python
eval_counter = {"n": 0}
seed_offset  = seed * 10_000_000

def fitness_fn(theta):
    eval_counter["n"] += 1
    rng_sim = np.random.default_rng(seed_offset + eval_counter["n"])
    return simulate_navigation(theta, m, arena, rng_sim, t_sim)
```

Counter persistent qua closure → mỗi fitness call có seed unique nhưng deterministic theo EA seed.

### 5.6. Solvers

Y hệt S1/S2 (9 EAs).

---

## 6. CLI Usage

```bash
# Full paper spec
python Scenario_3_run.py --n_rep 30 --n_evals 10000 --cores 4

# Smoke test
python Scenario_3_run.py --quick

# Chỉ Small-3 với 5 reps
python Scenario_3_run.py --problems ea.p.n.Small-3 --n_rep 5
```

**Args:**
- `--n_evals` (default 10000), `--n_rep` (default 30), `--cores` (default cpu_count-1).
- `--problems`: list subset.
- `--quick`: 2 reps, 300 evals, t_sim=10s.

---

## 7. Output format (CSV)

Giống S1/S2. Cột `problem` dạng `ea.p.n.{arena}-{m}`, `genotype_size` ∈ {122, 212, 464}.

---

## 8. Pipeline phân tích

Notebook `Scenario_3_local.ipynb`:
1. Đọc CSV.
2. Convergence plot (median + quantiles).
3. **NER per-problem**: rank ascending (minimize).
4. **EtTQ**: evals to Q3 threshold.
5. **NoVS pairwise**: Wilcoxon 2-sided α=0.01 cho mọi cặp solver.
6. **NER vs p**: log-scale x axis vì p ∈ {122, 212, 464}.
7. **NoVS per problem bar chart**.

---

## 9. Mapping sang Paper

| Paper | Code |
|-------|------|
| Physics (arena, robot, timestep) | Hằng số module, `simulate_navigation` |
| 3 arenas Figure 1 | `ARENAS` dict |
| m proximity sensors on frontal arc | `sensor_angles_local = linspace(-π/2, π/2, m)` |
| Target at (0.5, 0.15) | `TARGET` |
| ANN input = m + 2 | `n_in = m + 2` |
| Hidden = 3(m+2) | `n_hid = 3 * n_in` |
| Stochastic fitness | `eval_counter` + `seed_offset` |
| Fitness = mean distance | `total_dist / steps` |
| Figure 7, 8 | Notebook cells |
| Table 1 — 2D nav column | `_NOVS.xlsx` |

---

## 10. Khác biệt so với paper

**Paper dùng Java simulator (JGEA)**, tôi viết lại Python:

- **Rủi ro nhỏ**: Thuần vật lý (không dùng body dynamics, không có inertia riêng cho robot) → giả lập kinematic đơn giản. Paper cũng là kinematic (không nhắc tới dynamics).
- **Barrier collision**: paper không nêu chi tiết resolution. Code dùng "stay put if new_pos chạm barrier" — reasonable cho rigid circle vs segment.
- **Angle sensor**: paper viết "distance and angle to target" không rõ hình thức. Code dùng wrapped angle normalized ∈ [-1, 1], thay cho `sin(Δθ)` (ambiguous front/back).

Những khác biệt này giải thích vì sao NoVS ranking có thể lệch paper đôi chút (không phải bug, do simulator reimplementation).

---

## 11. Hiệu năng thực tế

- **Một episode 60s**: ~40ms (m=9, Maze arena).
- **10000 evals × 40ms**: ~400s/run.
- **Full run (9 EAs × 9 problems × 30 reps × 400s)** / 4 cores = **~67 giờ**.
- **Giảm 5 seeds × 3000 evals** = **~3.5 giờ** (đã chạy xong).

---

## 12. Kết quả thực nghiệm (5 seeds × 3000 evals)

**Mean NER overall (lower=better)**:
- ES-0.25: 0.272 (best)
- ES-0.5: 0.311
- ES-0.02: 0.378
- CMA-ES: 0.483
- GA variants: 0.51–0.58
- PSO: 0.678
- DE: 0.708 (worst)

**Lưu ý**: Paper có DE=9/9 NoVS cho S3 với 30 seeds. Với chỉ 5 seeds, stochastic noise của fitness chưa trung bình được, nên ranking không ổn định. Để so với paper cần ≥ 20 seeds.
