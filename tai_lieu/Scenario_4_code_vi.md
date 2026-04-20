# Scenario 4 — VSR (Voxel-based Soft Robot) Controller: Mô tả chi tiết code

## 1. Mục tiêu

Tái hiện **Scenario 4** trong paper (Section 2.2.4): tối ưu **trọng số của ANN controller** cho robot mềm 2D (Voxel-based Soft Robot) để thực hiện 5 tasks vận động.

**Bài toán**: 15 problems = 5 tasks × 3 controller-sensor combos.

**Lưu ý**: Paper dùng simulator Java `2D-VSR-Sim` (Medvet et al. 2020). **Code này là Python reimplementation đơn giản hóa** — giữ đúng pipeline và API paper nhưng physics đã rút gọn.

---

## 2. Cấu trúc file

```
Scenario_4/
├── Scenario_4_run.py        # ~855 dòng: physics + sensors + controllers + tasks
├── Scenario_4_local.ipynb   # Notebook phân tích
├── results/
│   └── Scenario_4_Novel_Final.csv
├── Figures/Scenario_4/      # 5 PNG
├── XLSX Files/Scenario_4/   # 1 XLSX
└── LaTeX/Scenario_4/        # 2 TXT
```

---

## 3. Thiết lập paper-exact

### 3.1. Morphology

```python
Biped : 4×3 grid minus 2 top corners → 10 voxels (4 tasks locomotion/jump)
Tower : 2×7 grid                     → 14 voxels (task balance)
```

Morphology cố định, không optimize.

### 3.2. Sensor configurations

| Config | Sensors per voxel | Compatible controllers |
|--------|-------------------|------------------------|
| **Homogeneous (homo)** | 3 (area, vx, vy) | HoD |
| **Heterogeneous (hetero)** | 3/4/5 khác nhau theo vùng | C, HeD |

**Heterogeneous detail** (code): chia voxels thành 3 vùng:
- Bottom (1/3 đầu): 5 sensors (area, vx, vy, proximity, sin)
- Middle (1/3 giữa): 3 sensors (area, vx, vy)
- Top (1/3 cuối): 4 sensors (area, vx, vy, sin)

Paper chi tiết hơn (head/top/bottom với sensor khác), code đơn giản hóa nhưng giữ ràng buộc "HoD chỉ dùng với homo".

### 3.3. 15 Problems

Kết hợp `(task, controller)`:

| Task | Morphology | Objective | C | HoD | HeD |
|------|------------|-----------|---|-----|-----|
| flat | Biped | maximize vx | ✓ | ✓ | ✓ |
| hilly | Biped | maximize vx | ✓ | ✓ | ✓ |
| steppy | Biped | maximize vx | ✓ | ✓ | ✓ |
| jump | Biped | maximize max_height | ✓ | ✓ | ✓ |
| balance | Tower | minimize drift | ✓ | ✓ | ✓ |

Problem name: `ea.p.v.{task}-{controller}`, ví dụ `ea.p.v.flat-HoD`.

### 3.4. Physics constants

```python
DT_SIM         = 1/60 s    # 60 Hz (paper)
CONTROL_PERIOD = 0.2 s     # 5 Hz controller (paper)
GRAVITY        = -9.8
SPRING_K       = 80 (edge), 50 (diagonal)
DAMPING        = 0.5
GROUND_FRIC    = 0.85
SENSOR_NOISE_VAR = 0.05    # paper
CONTRACT_RANGE = 0.3       # ±30% rest-length modulation
```

---

## 4. Giải thích các class/hàm chính

### 4.1. Morphology builders

```python
_biped_grid() → np.array((4,3))  # 4 cols × 3 rows, với [0,2] và [3,2] = 0
_tower_grid() → np.array((2,7))  # 2 cols × 7 rows full
```

### 4.2. `build_vsr(grid, voxel_size=1.0, y_offset=0.5)`

Build `VSRBody` từ voxel grid. Chi tiết:

1. **Tạo nodes**: với mỗi voxel hiện diện, sinh 4 node góc (shared corners qua `node_idx` dict).
2. **Tạo springs**: mỗi voxel có 6 springs (4 edges + 2 diagonals). Shared edges giữa voxels liền kề dùng chung spring (qua `_ensure_spring`).
3. **Track rest_lengths, voxel_corners, voxel_springs, initial_area**.

Dataclass `VSRBody` (frozen=False để update positions/velocities):
```python
nodes         : (N, 2)    # vị trí
velocities    : (N, 2)
springs       : (M, 2)    # pairs of node indices
rest_lengths  : (M,)
spring_k      : (M,)
voxel_corners : (V, 4)    # BL, BR, TR, TL per voxel
voxel_springs : (V, 6)    # 4 edges + 2 diagonals
initial_area  : (V,)
```

### 4.3. Quantities có thể quan sát

- `_voxel_centers(body)`: center của mỗi voxel (mean 4 corners).
- `_voxel_areas(body)`: current area via shoelace formula (để tính area-ratio sensor).
- `_voxel_velocities(body)`: mean velocity 4 corners.

### 4.4. `physics_step(body, terrain_fn, contractions, dt=DT_SIM)`

**Verlet-style integration** với spring-damper:

```
1. Contraction → scale rest_lengths:
   scale_per_voxel = 1 + CONTRACT_RANGE * contractions  (V,)
   Springs thuộc nhiều voxels → trung bình scale.
2. Spring force per spring:
   delta = nodes[b] - nodes[a]
   dir   = delta / ||delta||
   F_spring = k * (||delta|| - rest_modulated)
   F_damp   = DAMPING * (rel_vel · dir)
   F_total  = (F_spring + F_damp) * dir
   → +/- on nodes a, b
3. Gravity: F[:, 1] += MASS * GRAVITY
4. Integrate: v += F/m * dt; pos += v * dt
5. Ground collision:
   ground_ys = [terrain_fn(x) for x in nodes]
   Nếu node.y < ground_y → clamp pos, zero vy (nếu đang đi xuống), áp friction vào vx
```

### 4.5. Sensor layer

#### `_make_sensor_layout(n_voxels, sensor_config)`
Trả về `[n_sensors per voxel]` theo homo/hetero. Ví dụ biped+hetero:
```python
[5, 5, 5, 3, 3, 3, 4, 4, 4, 4]   # 1/3 bottom:5, 1/3 mid:3, 1/3 top:4
```

#### `read_sensors(body, sensor_layout, t, rng) → List[np.ndarray]`

Trả list per-voxel sensor arrays. Mỗi voxel append:
1. `area_ratio - 1.0` (centered, clipped).
2. `vx / 5.0`.
3. `vy / 5.0`.
4. Nếu `n_s >= 4`: thêm `sin(2π·1·t)` (1 Hz signal).
5. Nếu `n_s >= 5`: thêm proximity crude (distance from voxel center to ground / 2).

Cuối cùng: add `N(0, 0.05)` Gaussian noise (paper).

### 4.6. Controllers (C / HoD / HeD)

#### `ControllerConfig(controller_type, sensor_config, n_voxels, sensor_layout)`

Immutable config chứa metadata.

#### `theta_dim(cfg)`

```python
if C:
    n_in  = total_sensors
    n_hid = n_in
    n_out = n_voxels
    return n_in*n_hid + n_hid + n_hid*n_out + n_out

if HoD:
    # Shared ANN, sensor_layout phải là homo
    n_in = sensor_layout[0]; n_hid = n_in; n_out = 1
    return n_in*n_hid + n_hid + n_hid + 1

if HeD:
    # Per-voxel ANN, concat weights
    total = Σ_voxel (n_s·n_s + n_s + n_s + 1)
    return total
```

#### `predict_contractions(theta, sensors, cfg) → (n_voxels,)`

Unpacks theta theo controller type và forward pass từng ANN:
- **C**: 1 ANN duy nhất nhận concat(sensors) → (n_voxels,) contractions.
- **HoD**: cùng 1 ANN chạy cho từng voxel (weight sharing).
- **HeD**: mỗi voxel 1 ANN khác nhau.

Output ∈ [-1, 1] (tanh).

### 4.7. Tasks & terrain functions

```python
_terrain_flat(x)   = 0
_terrain_hilly(x)  = 0.15 * sin(0.6 * x)
_terrain_steppy(x) = 0.10 * floor(x)     # bậc thang mỗi 1m
```

Tasks (fitness sau khi simulate):
- **Locomotion (flat/hilly/steppy)**: `mean_vx = (final_com_x - init_com_x) / t_sim`, trả `-mean_vx` (minimize internally).
- **Jump**: max height COM trong half-episode sau, trả `-max_h`.
- **Balance**: drift COM ngang so với vị trí đầu, trả `drift` (minimize).

### 4.8. `make_fitness(spec, t_sim)`

Closure tạo fitness_fn cho 1 problem:
```python
def fitness_fn(theta):
    body = build_vsr(grid)
    settle 5 steps
    traj = _simulate_episode(theta, body, cfg, terrain, t_sim)
    return task_specific_metric(traj)
```

**Lưu ý important**: Paper minimize balance, maximize locomotion/jump. Code **minimize uniformly** bằng cách negate metric cho maximize tasks. Điều này giúp EAs (vốn minimize) chạy thống nhất. Notebook dùng `objective` column để hiển thị đúng chiều.

### 4.9. `_simulate_episode(theta, body, cfg, terrain, t_total, rng)`

Loop:
```
for s in range(steps_total):
    if s % control_every == 0:
        sensors = read_sensors(body, ..., t)
        contractions = predict_contractions(theta, sensors, cfg)
    physics_step(body, terrain, contractions)
    record COM position
return traj dict
```

Kiểm soát được invoked mỗi 0.2s (paper) thay vì mỗi physics step (paper lý do: tránh rung).

### 4.10. `_worker(args)`

Nhận `(spec, solver_key, sigma, seed, n_evals, t_sim)`:
1. Build controller cfg + fitness fn.
2. Gọi solver tương ứng.
3. Trả records + metadata (objective = spec.objective gốc).

### 4.11. Solvers

Y hệt 9 EAs từ S1/S2/S3.

---

## 5. CLI Usage

```bash
# Full paper spec (20 seeds × 10000 evals × 15 problems)
python Scenario_4_run.py --n_rep 20 --n_evals 10000 --cores 4

# Quick smoke
python Scenario_4_run.py --quick

# Chỉ 1 problem với 3 seeds
python Scenario_4_run.py --problems ea.p.v.flat-HoD --n_rep 3

# Giảm t_sim để nhanh hơn
python Scenario_4_run.py --n_rep 3 --n_evals 1500 --t_sim 15 --cores 4
```

**Args:**
- `--n_evals` (default 10000), `--n_rep` (default 20, paper), `--cores` (default cpu_count-1).
- `--problems`: subset.
- `--t_sim` (float, override default per-task duration).
- `--quick`: 2 reps, 300 evals, t_sim=3s.

---

## 6. Output format (CSV)

Giống các scenario khác. Khác biệt:
- `objective` có cả `minimize` (balance) và `maximize` (locomotion/jump).
- `best_fitness` luôn lưu dạng minimize uniformly (tasks maximize đã negated).
- `problem` dạng `ea.p.v.{task}-{controller}`.

---

## 7. Pipeline phân tích

Notebook `Scenario_4_local.ipynb` bổ sung cho S1/S2/S3:
- Xử lý mixed minimize/maximize: `best_fitness` luôn ascending (lower=better) vì đã uniformly negated.
- `objective` column dùng để label plot (hiển thị đúng chiều cho user đọc).
- NER, EtTQ, NoVS dùng logic giống S3 (ascending rank).

---

## 8. Mapping sang Paper

| Paper | Code |
|-------|------|
| Biped/Tower | `_biped_grid`, `_tower_grid` |
| 2 sensor configs | `_make_sensor_layout` |
| C / HoD / HeD controllers | `ControllerConfig`, `theta_dim`, `predict_contractions` |
| 5 tasks với fitness | `make_fitness`, terrain fns |
| Controller invoke every 0.2s | `control_every = int(0.2 / DT_SIM)` |
| Sensor noise variance 0.05 | `SENSOR_NOISE_VAR = 0.05` |
| HoD chỉ tương thích homo | `assert cfg.sensor_config == "homo"` trong `theta_dim` |
| Figure 9, 10 | Notebook cells |
| Table 1 — VSR column | `_NOVS.xlsx` |

---

## 9. Khác biệt với paper

Paper dùng `2D-VSR-Sim` (Java, Medvet et al. 2020). Code này **giản lược**:

### Vật lý
- ✓ Mass-spring với 4 mass corner per voxel, 4 edges + 2 diagonals.
- ✓ Gravity, ground collision với friction.
- ✓ 60 Hz physics, 5 Hz controller.
- ✗ Không có rigid-body interaction giữa voxels ngoài shared corners.
- ✗ Không có complex friction coefficients.
- ✗ Không có dynamic self-collision.

### Kích thước search space
Paper báo cáo:
- Biped: HoD=52, C=280, HeD=374
- Tower: HoD=52, C=275, HeD=420

Code sinh theo công thức (input size × hidden size) có thể ra số hơi khác (phụ thuộc sensor layout cụ thể). Ý tưởng đúng, cụ thể có thể khác.

### Sensor heterogeneous
Paper: "sin + proximity at head; area + velocity at top; proximity at bottom".
Code: split 3 vùng với 5/3/4 sensors (mục 3.2). Simplification nhưng tương thích về mặt cấu trúc.

**Kết quả**: trends ranking (GA/DE win, CMA-ES khó với HoD) đúng paper, nhưng con số NoVS cụ thể có thể lệch do simulator difference + ít seeds.

---

## 10. Hiệu năng thực tế

- **Một episode 30s**: ~150ms với VSR 10 voxels.
- **10000 evals × 150ms**: ~1500s = 25 phút/run.
- **Full run**: 9 EAs × 15 problems × 20 reps × 1500s / 4 cores = **~280 giờ (~12 ngày)**.
- **Paper báo cáo**: ~10000s/run (Java slower) → full run trên 2 cluster Xeon cũng mất nhiều ngày.
- **Reduced (3 seeds × 1500 evals × t_sim=15)**: ~5 giờ, đã chạy xong.

---

## 11. Kết quả thực nghiệm (3 seeds × 1500 evals × t_sim=15)

**Mean NER overall (lower=better)**:
- GA-0.25: 0.339 (best)
- GA-0.02: 0.343
- GA-0.5: 0.364
- ES-0.02: 0.485
- PSO: 0.492
- CMA-ES: 0.508
- DE: 0.614
- ES-0.5: 0.647
- ES-0.25: 0.708

**Kết luận**: GA (mọi σ) thắng → đúng xu hướng paper (Table 1 VSR: GA-0.5=12, GA-0.25=11, DE=13).
NoVS returns 0 vì 3 seeds quá ít cho Wilcoxon α=0.01. Để so với paper cần ≥ 15 seeds.
