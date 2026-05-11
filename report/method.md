# Methods (Phương pháp) — Simple Evolutionary Algorithm (EA) cho Neuroevolution

Tài liệu này mô tả phần **Methods** của đồ án: các thuật toán được so sánh, thiết lập thực nghiệm, cách triển khai, cách đếm ngân sách đánh giá, và cách ghi nhận kết quả. Nội dung bám sát paper (El Saliby et al., 2024) và code trong repo (các file `Scenario_1/Scenario_1_run.py` … `Scenario_4/Scenario_4_run.py`), đồng thời tương thích với ký hiệu trong `tai_lieu/algorithms.md` và `tai_lieu/scenarios_vi.md`.

---

## 1) Tổng quan phương pháp

### 1.1. Bài toán tối ưu hộp đen

Ở cả 4 scenario, ta tối ưu một vector thực \(\mathbf{x}\) hoặc \(\boldsymbol{\theta}\in\mathbb{R}^p\) để làm tốt một hàm fitness \(f(\cdot)\) (tính trực tiếp, hoặc qua ANN/mô phỏng). Vì \(f\) không có đạo hàm và có thể nhiễu/stochastic, ta dùng **evolutionary algorithms (EA)**.

### 1.2. Lý do dùng EA

- **Không cần gradient**: chỉ cần so sánh fitness.
- **Tổng quát**: áp dụng được cho hàm benchmark lẫn điều khiển qua mô phỏng.
- **Khả năng khám phá**: đặc biệt quan trọng khi landscape gồ ghề (neuroevolution).

### 1.3. Thiết kế thí nghiệm “so sánh công bằng”

Để so sánh thuật toán một cách fair:

- **Không gian tìm kiếm**: \(\mathbb{R}^p\), khởi tạo đồng nhất cho mọi EA: \(\mathcal{U}([-1,1]^p)\).
- **Ngân sách**: dừng theo số lần gọi fitness evaluation \(n_{\text{eval}}=10\,000\).
- **Lặp lại nhiều lần**: nhiều seeds (paper: 30 seeds cho S1–S3, 20 seeds cho S4).
- **Cùng hyperparameters như paper** (paper-exact), tránh “tuning” thiên vị.

---

## 2) Thiết lập chung dùng cho mọi scenario

### 2.1. Khởi tạo quần thể/điểm ban đầu

- Với các EA theo population (DE/PSO/ES/GA): quần thể ban đầu có kích thước \(n_{\text{pop}}\) và mỗi cá thể \(\sim \mathcal{U}([-1,1]^p)\).
- Với CMA-ES: điểm khởi tạo \(x_0\sim \mathcal{U}([-1,1]^p)\), step-size khởi tạo \(\sigma_0=0.5\).

### 2.2. Đếm “fitness evaluations” và điều kiện dừng

Một **fitness evaluation** là một lần gọi \(f(\mathbf{x})\) hoặc \(f(\boldsymbol{\theta})\). Mọi EA đều chạy đến khi tổng số evaluation **chạm/vượt** \(n_{\text{eval}}\) (mặc định 10,000).

Trong code, mỗi solver tự tăng bộ đếm `total_evals` theo số lần gọi fitness:

- CMA-ES: mỗi vòng `ask()` trả về \(\lambda\) nghiệm ⇒ tăng \(+\lambda\).
- PSO: mỗi iteration đánh giá lại toàn bộ \(n_{\text{pop}}\) hạt ⇒ tăng \(+n_{\text{pop}}\).
- ES/GA: mỗi generation đánh giá lại toàn bộ \(n_{\text{pop}}\) cá thể ⇒ tăng \(+n_{\text{pop}}\).
- DE: mỗi cá thể tạo một trial và đánh giá 1 lần ⇒ tăng \(+NP\) theo một “generation”.

### 2.3. Chuẩn hoá mục tiêu minimize/maximize

Trong S4 có bài toán maximize. Triển khai trong code dùng quy ước:

- **luôn trả về một giá trị để minimize**,
- với task maximize thì dùng \(\tilde f(\theta)=-f(\theta)\).

Điều này cho phép tái sử dụng cùng các solver “minimize-oriented”.

---

## 3) Bộ thuật toán so sánh (9 EA instances)

Repo so sánh 9 cấu hình EA (đúng paper), gồm 5 họ thuật toán nhưng ES/GA mỗi họ có 3 cấu hình theo \(\sigma\).

### 3.1. CMA-ES (Covariance Matrix Adaptation ES)

- **Ý tưởng**: học phân phối Gaussian để lấy mẫu nghiệm; cập nhật mean/covariance theo các nghiệm tốt nhất.
- **Thiết lập**:
  - `x0 ~ U([-1,1]^p)`
  - `sigma0 = 0.5`
  - các tham số còn lại dùng mặc định “vanilla” của thư viện `cma` (phù hợp mô tả paper).
- **Mỗi iteration**:
  - `solutions = es.ask()` (lấy mẫu \(\lambda\) cá thể)
  - đánh giá fitness cho từng cá thể
  - `es.tell(solutions, fits)` để cập nhật.

### 3.2. DE/rand/1/bin (Differential Evolution)

- **Hyperparameters (paper-exact)**: \(NP=15\), \(F=0.5\), \(CR=0.8\).
- **Mutation**:
  \[
  \mathbf{v}_i=\mathbf{x}_{r_1}+F(\mathbf{x}_{r_2}-\mathbf{x}_{r_3})
  \]
- **Crossover** theo từng chiều (binomial): lấy từ \(\mathbf{v}\) với xác suất \(CR\), đảm bảo ít nhất 1 chiều lấy từ mutant.
- **Selection 1–1**: nếu trial tốt hơn thì thay target.

### 3.3. PSO (Particle Swarm Optimization, Clerc 2012-style)

- **Hyperparameters (paper-exact)**: \(n_{\text{pop}}=100\), \(\omega=0.8\), \(\phi_p=\phi_g=1.5\).
- **Cập nhật**:
  \[
  \mathbf{v}\leftarrow \omega\mathbf{v}+\phi_p r_1(\mathbf{p}-\mathbf{x})+\phi_g r_2(\mathbf{g}-\mathbf{x}),
  \quad
  \mathbf{x}\leftarrow \mathbf{x}+\mathbf{v}.
  \]
- **Personal best / global best** được cập nhật sau mỗi vòng đánh giá.

### 3.4. ES (basic) — ES-\(\sigma\) với 3 cấu hình

- **Hyperparameters (paper-exact)**:
  - \(n_{\text{pop}}=30\)
  - \(n_{\text{parents}}=\lfloor 0.33n_{\text{pop}}\rfloor=10\)
  - elitism: giữ lại best cá thể của thế hệ trước
  - \(\sigma\in\{0.02,0.25,0.5\}\) ⇒ ES-0.02 / ES-0.25 / ES-0.5
- **Cập nhật**:
  - sắp theo fitness, lấy top parents
  - tính mean \(\boldsymbol{\mu}\)
  - sinh \(n_{\text{pop}}-1\) cá thể mới:
    \[
    \mathbf{x}'=\boldsymbol{\mu}+\epsilon,\quad \epsilon\sim\mathcal{N}(0,\sigma I).
    \]

**Lưu ý triển khai quan trọng**: trong code, \(\sigma\) được hiểu là **variance**, nên chuẩn sai là \(\sqrt{\sigma}\) khi lấy mẫu Gaussian.

### 3.5. GA số thực — GA-\(\sigma\) với 3 cấu hình

- **Hyperparameters (paper-exact)**:
  - \(n_{\text{pop}}=100\)
  - tournament size \(n_{\text{tour}}=5\)
  - xác suất crossover \(p_{\text{xo}}=0.8\)
  - \(\sigma\in\{0.02,0.25,0.5\}\) ⇒ GA-0.02 / GA-0.25 / GA-0.5
- **Selection**: tournament selection.
- **Variation**:
  - Nếu crossover:
    \[
    \mathbf{x}'=\mathbf{x}_1+\alpha(\mathbf{x}_2-\mathbf{x}_1)+\epsilon,\ 
    \alpha\sim U([0,1]),\ \epsilon\sim\mathcal{N}(0,\sigma I).
    \]
  - Nếu mutation:
    \[
    \mathbf{x}'=\mathbf{x}+\epsilon,\quad \epsilon\sim\mathcal{N}(0,\sigma I).
    \]
- **Replacement**: merge \((P \cup \text{Offspring})\) rồi giữ top \(n_{\text{pop}}\) (elitism theo kiểu “merge + truncate”).

---

## 4) Cách triển khai fitness theo từng scenario (mức methods)

Phần formulation đã định nghĩa \(f(\cdot)\); phần Methods nhấn mạnh cách **tính fitness trong code** và đặc điểm (deterministic/stochastic).

### 4.1. Scenario 1 — Synthetic

- Fitness được tính trực tiếp theo công thức (Sphere/PA/CPA/Ackley/Rastrigin).
- Deterministic: cùng một \(\mathbf{x}\) cho cùng fitness.
- CPA có tập 5 điểm mục tiêu cố định theo dimension (tạo bằng seed cố định) để tái lập.

### 4.2. Scenario 2 — Regression

- Dataset tải từ UCI và được cache dưới dạng `.npz` đã preprocess:
  - chuẩn hoá \(X\),
  - rescale \(y\) về \([-1,1]\).
- Fitness là MSE trên toàn bộ dataset (deterministic).
- Genotype \(\boldsymbol{\theta}\) được unpack thành \(W_1,b_1,W_2,b_2\) và chạy forward với `tanh`.

### 4.3. Scenario 3 — 2D Navigation

- Fitness là **stochastic**: mỗi lần gọi fitness chạy mô phỏng với khởi tạo ngẫu nhiên (vị trí/heading).
- Để vừa stochastic vừa tái lập, code dùng một `eval_counter` tăng dần và seed mô phỏng = `seed_offset + eval_counter`.
- Một fitness evaluation (paper-style) dùng **1 episode** (1 rollout) và trả về mean distance robot–target.

### 4.4. Scenario 4 — VSR Control

- Mô phỏng mass-spring 2D rút gọn để giữ “pipeline” như paper.
- Controller invoked mỗi 0.2s, physics step 60Hz, sensor noise Gaussian var 0.05.
- Fitness được quy về minimize:
  - locomotion/jump: trả về \(-\text{metric}\) để maximize metric.
  - balance: trả về drift/góc proxy để minimize.

---

## 5) Thiết lập thực nghiệm (runs, problems, parallelism)

### 5.1. Số problems và naming scheme

- S1: 24 problems (6 functions × 4 dims).
- S2: 9 problems (3 dataset × 3 kiến trúc \(\rho_h\)).
- S3: 9 problems (3 arena × \(m\in\{3,5,9\}\)).
- S4: 15 problems (5 tasks × 3 combo controller/sensor).

Trong code, mỗi problem có `problem_name` theo dạng `ea.p.<scenario>.<name>` và được ghi vào CSV.

### 5.2. Số seeds (repetitions)

- S1–S3: \(n_{\text{rep}}=30\) (mặc định).
- S4: \(n_{\text{rep}}=20\) (mặc định, do mô phỏng nặng hơn).

### 5.3. Song song hoá

- Dùng multiprocessing kiểu `spawn` để tương thích Windows.
- Mỗi task tương ứng một tuple: (problem, solver, sigma nếu có, seed, budget, …).
- Với S2, dataset được load theo-process để tránh pickle mảng lớn.

### 5.4. Chế độ “quick / smoke test”

Mỗi scenario có `--quick` để kiểm tra pipeline nhanh:

- giảm `n_rep`, giảm `n_evals`,
- thường chạy 1 core,
- rút ngắn thời gian mô phỏng ở scenario có simulator.

---

## 6) Ghi log và xuất kết quả (output schema)

### 6.1. Record theo mốc iteration

Các solver ghi log định kỳ mỗi 5 iteration (và/hoặc iteration 0). Mỗi record gồm:

- `iterations`: số iteration hiện tại
- `evals`: tổng số fitness evals đã dùng
- `births`: số cá thể mới sinh (để theo dõi mức biến thiên)
- `elapsed`: thời gian chạy (giây)
- `best_fitness`: best (min) trong population ở thời điểm log
- `geno_uni`, `fit_uni`: độ đa dạng genotype/fitness (xấp xỉ bằng unique ratio sau khi làm tròn)

### 6.2. File CSV kết quả

Mỗi scenario xuất một file:

- `Scenario_1/results/Scenario_1_Novel_Final.csv`
- `Scenario_2/results/Scenario_2_Novel_Final.csv`
- `Scenario_3/results/Scenario_3_Novel_Final.csv`
- `Scenario_4/results/Scenario_4_Novel_Final.csv`

Định dạng:

- phân tách bằng dấu `;`
- chứa các cột: `seed`, `problem`, `solver_sigma`, `objective`, `best→fitness`, `genotype_size`, … cùng các trường record ở trên.

---

## 7) Liên hệ với phần đánh giá (metrics)

Các file CSV “per-iteration” là đầu vào để notebook tổng hợp:

- final best fitness (lấy bản ghi cuối/hoặc min theo `evals`),
- NER/EtTQ/NoVS (tính theo paper).

Trong báo cáo, phần Methods kết thúc tại đây; phần “Results/Discussion” sẽ dùng các metric trên để trả lời câu hỏi nghiên cứu: *khi nào EA đơn giản là đủ, và vai trò của crossover trong neuroevolution là gì?*

