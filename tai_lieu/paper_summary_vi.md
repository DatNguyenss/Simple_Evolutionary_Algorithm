# Tóm tắt chi tiết bài báo (tiếng Việt)

> **Bài báo gốc**: Michel El Saliby, Giorgia Nadizar, Erica Salvato, Eric Medvet (2024).
> *Eventually, All You Need is a Simple Evolutionary Algorithm (for Neuroevolution of Continuous Control Policies).*
> GECCO '24 Companion, July 14–18, 2024, Melbourne, Australia. DOI: 10.1145/3638530.3664112
>
> **Lưu ý bản quyền**: Đây là tài liệu tóm tắt–diễn giải bằng tiếng Việt phục vụ mục đích học tập và nghiên cứu cho đồ án môn Nhập môn Machine Learning. Không phải bản dịch toàn văn. Mọi công thức, bảng số liệu và thuật ngữ chuyên môn giữ nguyên như trong bản gốc. Khi trích dẫn trong báo cáo, trích đúng từ bản gốc tiếng Anh.

---

## Thông tin xuất bản

- **Tác giả**: Michel El Saliby, Giorgia Nadizar, Erica Salvato, Eric Medvet (Đại học Trieste, Ý).
- **Hội nghị**: GECCO '24 Companion (Genetic and Evolutionary Computation Conference Companion), 14–18/7/2024.
- **Nhà xuất bản**: ACM. ISBN: 979-8-4007-0495-6/24/07.
- **Loại bài**: Research article, 10 trang (1904–1913 trong kỷ yếu).
- **Repo mã nguồn**: https://github.com/elsalibymichel/2024_NEWK-GECCO_EA-Comparison
- **Framework dùng**: JGEA (Java Genetic and Evolutionary Algorithms).

---

## Tóm tắt (Abstract)

Bài báo giải bài toán tối ưu hóa trọng số của **mạng nơ-ron nhân tạo (ANN)** với kiến trúc cố định bằng **thuật toán tiến hóa (EA)** — đây là một dạng **neuroevolution**. Các tác giả so sánh hiệu năng của nhiều EA trên các bài toán điều khiển liên tục (continuous control), cụ thể là:

- Bài toán điều hướng 2D cho robot bánh xe.
- Bài toán điều khiển robot mềm dạng voxel (Voxel-based Soft Robots, VSR).
- Bài toán hồi quy (regression) dùng ANN (làm tham chiếu).
- Bài toán tối ưu số học chuẩn (benchmark) như Sphere, Ackley, Rastrigin (làm tham chiếu).

**Phát hiện chính**: Các EA đơn giản như **Genetic Algorithm (GA)** và **Differential Evolution (DE)** đạt hiệu năng tốt trên bài toán điều khiển, **mặc dù** chúng bị các thuật toán tinh vi hơn (như CMA-ES) đánh bại trên các benchmark toán học.

**Giả thuyết**: Hiệu quả của các EA đơn giản đến từ **toán tử crossover (lai ghép)** — toán tử này có lợi trên các **fitness landscape gồ ghề (rugged)** xuất hiện trong bài toán điều khiển phức tạp.

**Từ khóa**: Neuroevolution, Continuous control, Policy search.

---

## 1. Giới thiệu và các công trình liên quan

### 1.1. Bối cảnh

EA là công cụ tối ưu hóa mạnh, được dùng rộng rãi trong nhiều lĩnh vực. Khi áp dụng EA để tiến hóa ANN, ta gọi là **neuroevolution**. Các bài toán điều khiển liên tục là nhóm thách thức đặc biệt vì không gian tìm kiếm **phức tạp và nhiều chiều**.

EA đã được chứng minh là giải pháp thay thế hứa hẹn so với:
- **Data-driven control** (các phương pháp điều khiển dựa dữ liệu).
- **Reinforcement learning (RL)** — học tăng cường.

### 1.2. Khoảng trống mà bài báo lấp vào

Các nghiên cứu trước đó so sánh các bộ điều khiển ANN tiến hóa:
- Chỉ tập trung vào **một miền cụ thể** (Atari games, fault diagnosis, robot navigation).
- Thường **so sánh từng cặp** (pairwise), khiến việc đánh giá xu hướng chung bị hạn chế.
- Một số công trình so sánh neuroevolution với RL, nhưng cuộc tranh luận này đã kéo dài nhiều thập kỷ.

**Câu hỏi mở**: *Liệu một EA tinh vi có thực sự cần thiết để giải neuroevolution trên bài toán điều khiển phức tạp, hay các EA đơn giản cũng đủ?*

### 1.3. Đóng góp của bài báo

Bài báo thực hiện **phân tích so sánh thấu đáo** hiệu năng của **5 EA nổi bật**:

1. **CMA-ES** — Covariance Matrix Adaptation Evolution Strategy.
2. **DE** — Differential Evolution.
3. **PSO** — Particle Swarm Optimization.
4. **ES** — Evolutionary Strategy (biến thể đơn giản).
5. **GA** — Genetic Algorithm.

Áp dụng trên:
- 2D robot navigation.
- Modular soft robot control.
- Regression problems (tham chiếu).
- Synthetic benchmark problems (tham chiếu).

**Metric đánh giá**:
- **Search effectiveness** (hiệu quả tìm kiếm) — đo bằng **NER** (Normalized Effectiveness Rank) của lời giải tốt nhất ở iteration cuối.
- **Search efficiency** (hiệu suất tìm kiếm) — đo bằng số lần đánh giá fitness cần để đạt mức chất lượng thỏa đáng.

### 1.4. Kết quả chính (nêu sơ bộ)

- Ngay cả các EA đơn giản nhất như GA và DE đạt hiệu năng **đáng khen** trên task achievement.
- Trên benchmark, chúng không so được với các phương pháp tinh vi; **nhưng** trên control tasks, hiệu năng của chúng đáng chú ý.
- Hai EA tốt nhất trên control problems (GA và DE) là **hai EA duy nhất dùng toán tử recombination (crossover)**.
- Crossover giúp khám phá hiệu quả các fitness landscape phức tạp → khiến EA đơn giản trở nên hiệu quả và hiệu suất cho neuroevolution.

---

## 2. Phương pháp (Methods)

### 2.1. Các thuật toán tiến hóa

Nghiên cứu xét **5 EA** nổi tiếng cho tối ưu số học: CMA-ES, DE, PSO, ES, GA. Không gian tìm kiếm là \( \mathbb{R}^p \), mỗi lời giải \( \mathbf{x} \in \mathbb{R}^p \) là vector số thực \(p\) chiều. Tất cả EA được thiết kế cho \( \mathbb{R}^p \), trừ GA (tổng quát hơn, có thể áp dụng cho không gian khác).

#### 2.1.1. CMA-ES

- Là dạng ES, sử dụng cơ chế **ước lượng phân phối** các tham số cần tối ưu.
- Có chiến lược **điều chỉnh step-size thích ứng** (độ lớn của bước đi trong \( \mathbb{R}^p \) giữa các iteration).
- Đã được dùng rộng rãi cho các bài toán điều khiển, bao gồm cả loại robot mô phỏng trong nghiên cứu này.
- Bài báo dùng **vanilla CMA-ES** theo Hansen (2016), **giữ nguyên toàn bộ tham số default**.

#### 2.1.2. DE (Differential Evolution)

- EA đơn giản cho tối ưu số học, **cải thiện lặp** candidate solutions bằng cách recombine chúng qua crossover.
- Có nhiều biến thể, thường đặt tên theo schema **DE/a/b/c**.
- Bài báo dùng **DE/rand/1** với các tham số:
  - **Population size**: n_pop = NP = **15**
  - **Differential weight**: w_diff = F = **0.5**
  - **Crossover probability**: p_xo = CR = **0.8**

#### 2.1.3. PSO (Particle Swarm Optimization)

- Phương pháp tối ưu lấy cảm hứng sinh học, nhưng không từ tiến hóa tự nhiên — mà từ ý tưởng **các hạt di chuyển trong không gian tìm kiếm**, được điều khiển bởi lực **local (cognitive)** và **global (social)**.
- Bài báo dùng phiên bản chuẩn của Clerc (2012) với tham số:
  - **n_pop = 100**
  - **Cognitive parameter**: w = **0.8**
  - **Cognitive acceleration**: φ_particle = **1.5**
  - **Social acceleration**: φ_global = **1.5**

#### 2.1.4. ES (Evolutionary Strategy) — biến thể cơ bản

CMA-ES cũng thuộc họ ES, nhưng để đánh giá một thành viên **đơn giản hơn**, bài báo cài biến thể cơ bản như sau:

- Duy trì quần thể **n_pop** candidate solutions (điểm trong \( \mathbb{R}^p \)).
- Mỗi iteration:
  1. Lấy **n_parents** best solutions (tốt nhất theo fitness).
  2. Tính **trung bình** \( \boldsymbol{\mu} \) của các parents.
  3. Tạo quần thể mới: giữ lại **best solution hiện tại** + \( (n_{\text{pop}} - 1) \) điểm mới sampled từ \( \mathcal{N}(\boldsymbol{\mu}, \sigma \mathbf{I}) \).
- Mỗi solution mới = áp dụng Gaussian noise zero-mean, phương sai σ cho từng thành phần.

**Tham số**:
- n_pop = **30**
- n_parents = \( \lfloor 0.33 \cdot n_{\text{pop}} \rfloor \) = **10**
- \( \sigma \in \{0.02, 0.25, 0.5\} \) → **3 biến thể ES**.

ES đơn giản này đã được dùng để giải control problems với modular soft robots, kể cả với recurrent và plastic neural controllers.

#### 2.1.5. GA (Genetic Algorithm)

Là EA tổng quát nhất trong nghiên cứu, EA duy nhất có thể áp dụng ngoài \( \mathbb{R}^p \). Bài báo dùng biến thể chuẩn tùy chỉnh cho numerical optimization:

**Một iteration**:
1. Từ quần thể hiện tại n_pop **parents**, tạo **offspring population** gồm n_pop solutions, bằng cách lặp n_pop lần:
   1. Chọn **crossover** (xác suất p_xo) hoặc **mutation** (xác suất 1 − p_xo).
   2. Chọn 1 (mutation) hoặc 2 (crossover) parents bằng **tournament selection** kích thước n_tour.
   3. Áp dụng operator để tạo solution mới.
2. Merge parents + offspring, giữ **n_pop solutions tốt nhất** cho iteration tiếp theo.

**Tham số**:
- n_pop = **100**, p_xo = **0.8**, n_tour = **5**.

**Genetic operators**:
- **Gaussian mutation**: offspring \( \mathbf{x}' \) từ parent \( \mathbf{x} \) theo công thức \( \mathbf{x}' = \mathbf{x} + \boldsymbol{\varepsilon} \), với \( \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma \mathbf{I}) \).
- **Segment geometric crossover**: offspring x' từ parents x₁, x₂ theo:

  \( \mathbf{x}' = \mathbf{x}_1 + \alpha(\mathbf{x}_2 - \mathbf{x}_1) + \boldsymbol{\varepsilon} \), với \( \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma \mathbf{I}) \) và \( \alpha \sim \mathcal{U}([0,1]) \).

- \( \sigma \in \{0.02, 0.25, 0.5\} \) → **3 biến thể GA**.

GA đơn giản này cũng đã được dùng tiến hóa neural controllers cho modular soft robots, kể cả với ANNs dùng attention mechanisms.

#### 2.1.6. Điều kiện dừng chung

- Tất cả EA đều **lặp**. Để so sánh fair, dùng **cùng điều kiện dừng**:
  - Dừng tại iteration đầu tiên mà số fitness evaluations vượt quá **n_eval = 10 000**.
- Khởi tạo quần thể ban đầu: sampling \( \mathbf{x} \sim \mathcal{U}([-1, 1])^p \) cho mọi EA.

#### 2.1.7. Ghi chú về step-size σ

Tham số \( \sigma \) trong ES và GA xác định **độ lớn của bước đi**:
- **σ lớn** → khám phá nhanh search space, nhưng khó focus vào vùng hứa hẹn.
- **σ nhỏ** → ngược lại.

σ đóng vai trò then chốt trong **trade-off exploration vs exploitation**, và rất khó chọn giá trị tối ưu. Vì vậy, bài báo **thử 3 giá trị σ** cho 2 EA tổng quát nhất có dùng σ là ES và GA.

**Ghi chú quan trọng**: GA và DE là **hai EA duy nhất** trong so sánh này có sử dụng **crossover thực sự**.

---

### 2.2. Các bài toán tối ưu số học

Bài báo chia bài toán thành **4 scenarios**:

- **Scenario 1**: synthetic — \( \mathbf{x} \) là điểm, tối ưu trực tiếp hàm \( f(\mathbf{x}) \).
- **Scenario 2, 3, 4**: \( \boldsymbol{\theta} \) là vector tham số ANN → neuroevolution.
  - Scenario 2: ANN là regressor, fitness = MSE (trực tiếp nhất).
  - Scenario 3: ANN điều khiển robot 2D (gián tiếp hơn).
  - Scenario 4: ANN điều khiển VSR (môi trường vật lý phức tạp nhất, gián tiếp nhất).

Các scenarios khác nhau ở **mức độ "trực tiếp"** của mapping \( \boldsymbol{\theta} \to f(\boldsymbol{\theta}) \). Đây là yếu tố chính bài báo muốn khảo sát ảnh hưởng đến hiệu năng EA.

---

#### 2.2.1. Scenario 1 — Synthetic problems

**Mục tiêu**: tạo bộ bài toán đa dạng về các đặc tính: unimodal/multimodal, regular/irregular, separable/non-separable, problem size p, khoảng cách từ optimum đến origin.

**6 hàm mục tiêu**, đều **minimize**, với 4 kích cỡ search space \( p \in \{20, 100, 200, 500\} \) → tổng **24 bài toán**.

| # | Tên | Công thức | Optimum \( \mathbf{x}^\star \) | Đặc tính |
|---|-----|-----------|-----------|----------|
| 1 | **Sphere** | \( f(\mathbf{x}) = \sum_i x_i^2 \) | \( \mathbf{0} \) | Unimodal, dễ nhất |
| 2 | **PA-1** (Point Aiming) | \( f(\mathbf{x}) = \lVert \mathbf{x} - \mathbf{x}^\star \rVert \) | \( \mathbf{x}^\star = \mathbf{1} \) | Unimodal |
| 3 | **PA-10** | \( f(\mathbf{x}) = \lVert \mathbf{x} - \mathbf{x}^\star \rVert \) | \( \mathbf{x}^\star = 10\mathbf{1} \) | Unimodal, optimum xa origin |
| 4 | **CPA** (Circular Point Aiming) | \( f(\mathbf{x}) = \min_{i \in \{1,\dots,5\}} \lVert \mathbf{x} - \mathbf{x}_i^\star \rVert \) | 5 điểm ngẫu nhiên trên vòng tròn tâm 1, bán kính 2 | Multimodal |
| 5 | **Ackley** | \( f(\mathbf{x}) = -20\exp\!\Big(-0.2\sqrt{\tfrac{1}{p}\sum_i x_i^2}\Big) - \exp\!\Big(\tfrac{1}{p}\sum_i \cos(2\pi x_i)\Big) + 20 + e \) | \( \mathbf{0} \) | Multimodal, non-separable |
| 6 | **Rastrigin** | \( f(\mathbf{x}) = 10p + \sum_i \big(x_i^2 - 10\cos(2\pi x_i)\big) \) | \( \mathbf{0} \) | Multimodal, rugged landscape |

**Ghi chú**:
- Ackley và Rastrigin đều có global optimum tại \( \mathbf{0} \), nhưng có **nhiều local optima** gây nhiễu cho search.
- Với CPA, 5 target points cố định cho mọi experiment với cùng \(p\).
- PA là generalization của Sphere khi optimum \(\neq 0\). Hai biến thể \( \mathbf{x}^\star = \mathbf{1} \) và \( \mathbf{x}^\star = 10\mathbf{1} \) đánh giá khả năng đối phó với optimum gần/xa origin (nhớ: EA khởi tạo quanh 0).

**Ký hiệu bài toán**: `name-p` (ví dụ: `Sphere-100`, `Rastrigin-500`).

---

#### 2.2.2. Scenario 2 — ANN-based regression

**Bài toán hồi quy**: với dataset \( D = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n \), \( \mathbf{x}^{(i)} \in \mathbb{R}^m \), \( y^{(i)} \in \mathbb{R} \), tìm hàm \( g: \mathbb{R}^m \to \mathbb{R} \) minimize **MSE**:

\[
f(g) = \mathrm{MSE}(g, D) = \frac{1}{n}\sum_i \big(g(\mathbf{x}^{(i)}) - y^{(i)}\big)^2
\]

**Cách cài đặt**:
- Dùng ANN làm hàm g, với m inputs và 1 output, **kiến trúc cố định**.
- Tối ưu weights θ của ANN bằng EA, fitness = MSE.
- **3 kiến trúc** đều là feed-forward fully-connected, **1 hidden layer**, khác nhau ở kích thước hidden layer: **m, 2m, 3m** (tức \( \rho_h \in \{1,2,3\} \)).
- Số tham số: \( p = (m+1)\rho_h m + (\rho_h m + 1)k \), với \(k\) = số output (ở đây \(k=1\)).
- Activation function: **tanh** cho mọi neuron (cả hidden và output).

**3 dataset** (chọn theo White et al. 2013):
- **Concrete**: m = 8, |D| = 825.
- **Energy Efficiency**: m = 8, |D| = 615.
- **Wine**: m = 11, |D| = 3919.

**Tiền xử lý**:
- Standardize features (x⁽ⁱ⁾).
- Rescale target y⁽ⁱ⁾ về [−1, 1], để ANN với tanh output có thể fit.

**Tổng**: 9 bài toán = 3 datasets × 3 kiến trúc. Kích thước search space từ **p = 81** (Concrete-1, Energy-1) đến **p = 430** (Wine-3).

**Ký hiệu**: `name-ρₕ` (ví dụ `Concrete-2`, `Wine-3`).

---

#### 2.2.3. Scenario 3 — ANN-based 2D navigation

**Setup**:
- Agent: robot differential-drive, hình tròn, 2 bánh.
- Mục tiêu: đến target position trong arena có obstacles.
- Arena: **1m × 1m**, robot bán kính **5 cm**, max speed **10 cm/s**.
- 2 output điều khiển vận tốc 2 bánh.
- Input: **m proximity sensors** (khoảng cách tới obstacles, range 1m, đọc 1 nếu không có obstacle trong range, 0 nếu chạm trực tiếp) phân bố đều ở nửa phía trước + 2 sensors đọc khoảng cách và góc tới target.

**Simulation**:
- Thời gian: **60 s**, time step \( \delta t = 0.1\,\text{s} \).
- Robot bắt đầu tại \( (x_0, y_0) \) với \( x_0 \sim \mathcal{U}([0.45, 0.55]) \), \( y_0 \sim \mathcal{U}([0.80, 0.85]) \) → fitness **stochastic**.
- Target cố định tại (0.50, 0.15).

**3 arena** (xem Figure 1 trong paper gốc):
- **Small**: 1 barrier nhỏ trước target.
- **Large**: barrier lớn hơn.
- **Maze**: 2 barriers tạo mê cung đơn giản.

**3 robot setup** khác nhau số sensors: **m ∈ {3, 5, 9}**.

**ANN**:
- Input: m + 2 neurons (sensors + 2 target info).
- Hidden: **3(m + 2)** neurons, single layer.
- Output: 2 neurons (2 wheel velocities).
- Activation: tanh.
- Kích thước search space: **p ∈ {122, 212, 464}**.

**Fitness**: khoảng cách trung bình giữa agent và target trong suốt simulation → **minimize**.

**Tổng**: 9 bài toán = 3 arenas × 3 robot setups. Ký hiệu: `name-m` (ví dụ `Maze-5`).

---

#### 2.2.4. Scenario 4 — ANN-based VSR control

**VSR** (Voxel-based Soft Robots) là họ robot mềm module gồm các khối đàn hồi (voxels) có thể co giãn để tạo chuyển động, giống cơ bắp. Bài báo dùng mô hình **2D** (voxels là hình vuông) theo simulator 2D-VSR-Sim (Medvet et al. 2020), mỗi voxel mô hình hóa bởi 4 mass tại góc nối bởi các spring-damper.

**Morphology (hình dạng)**:
- 2 morphologies:
  - **Biped** (10 voxels) — dùng cho locomotion và jump.
  - **Tower** (14 voxels) — dùng cho balance.
- Bài báo **không** tối ưu morphology, chỉ tối ưu controller.

**Sensors** (4 loại):
- **proximity**: khoảng cách tới vật gần nhất theo hướng \( \alpha \), range \( r \).
- **area**: tỷ lệ diện tích voxel hiện tại so với rest area.
- **velocity**: vận tốc voxel theo trục x hoặc y.
- **sin**: sensor "giả" phát tín hiệu sine tần số 1 Hz.

**2 cấu hình sensor**:
- **Homogeneous**: area + velocity (cả 2 trục) ở mỗi voxel.
- **Heterogeneous**: sin + proximity (\(r=5\,\text{m}\), \( \alpha=-\pi/12 \)) ở "đầu" VSR (voxel trên cùng bên phải), area + velocity ở hàng voxels trên cùng, proximity (\(r=1\,\text{m}\), \( \alpha=-\pi/2 \)) ở hàng voxels dưới.

Sensor output normalized về [−1, 1].

**Controllers**:

1. **Centralized (C)**: 1 ANN duy nhất nhận toàn bộ sensor readings, output contraction [−1, 1] cho mỗi voxel.
2. **Homo-distributed (HoD)**: mỗi voxel có ANN riêng **nhưng cùng** architecture và parameters; trao đổi data với hàng xóm. Chỉ tương thích với homogeneous sensors.
3. **Hetero-distributed (HeD)**: mỗi voxel có ANN **khác nhau**; search space = concatenation toàn bộ parameters của mọi ANN.

Kết hợp: C và HeD dùng với heterogeneous sensors; HoD dùng với homogeneous sensors.

**Cấu hình ANN chung**: 1 hidden layer cùng kích thước input layer, tanh activation, Gaussian noise phương sai 0.05 trên mỗi sensor reading. Invoke controller mỗi 0.2 s (không phải mỗi time step) để tránh rung; giữ output constant giữa các lần invoke.

**Kích thước search space**:
- **Biped**: HoD p=52, C p=280, HeD p=374.
- **Tower**: HoD p=52, C p=275, HeD p=420.

**Tasks** (xem Figure 2 trong paper gốc):

| Task | Mô tả | Fitness | Goal |
|------|-------|---------|------|
| **Flat locomotion** | Di chuyển trên địa hình phẳng | v_x trung bình của center of mass, sim 30s | Maximize |
| **Hilly locomotion** | Địa hình đồi | v_x trung bình, sim 30s | Maximize |
| **Steppy locomotion** | Địa hình bậc thang (piece-wise flat altitude thay đổi) | v_x trung bình, sim 30s | Maximize |
| **Jump** | VSR nhảy cao nhất có thể | Max height của center of mass so với ground, sim 10s, bỏ 5s đầu | Maximize |
| **Balance** | VSR đặt trên xích đu 30m có khớp ở giữa, giữ thăng bằng | Góc trung bình của xích đu, sim 30s, phạt khi xích đu chạm đất | Minimize |

Simulator time step: **1/60 s**. Biped dùng cho 4 tasks (trừ balance); Tower dùng cho balance.

**Tổng**: 15 bài toán = 5 tasks × 3 controllers. Ký hiệu: `task-controller` (ví dụ `Flat-C`, `Balance-HeD`).

---

### 2.3. Các chỉ số so sánh (Comparison indexes)

#### 2.3.1. Search effectiveness

**Metric gốc**: \( f(\mathbf{x}^\*) \) — fitness của best solution tại iteration cuối. Nhưng fitness giữa các bài toán và scenarios **không so sánh được trực tiếp** (khác scale, khác bản chất).

**NER — Normalized Effectiveness Rank**

Quy trình, cho 1 problem với n_EAs thuật toán:
1. Chạy mỗi EA **n_rep** lần, thu n_rep giá trị final best fitness.
2. Sort toàn bộ n_EAs × n_rep giá trị.
3. Gán rank: **1** cho tốt nhất, **n_EAs × n_rep** cho tệ nhất.
4. Normalize rank về [0, 1].

**NER ∈ [0, 1]**, **càng thấp càng tốt** (0 = run đó tốt nhất trong bài toán).

**NoVS — Number of Victories Score**

Cho 1 problem:
1. Xác định **best EA** = EA có **median \( f(\mathbf{x}^\*) \)** tốt nhất.
2. Một EA "ghi chiến thắng" trên problem đó nếu final fitness của nó **không khác biệt có ý nghĩa thống kê** với best EA, kiểm định bằng **Wilcoxon signed-rank test** với \(H_0\): hai sample có cùng mean, significance level \( \alpha = 0.01 \).
3. NoVS của một EA trong 1 scenario = số problems mà EA đó ghi chiến thắng.

**Lưu ý**: Wilcoxon paired → các EA phải chạy cùng danh sách seeds.

#### 2.3.2. Search efficiency

**EtTQ — Evals to Third Quartile**

Quy trình, cho 1 problem:
1. Tính **Q3 (quartile thứ 3)** của tất cả n_EAs × n_rep final fitness.
2. Với mỗi run, tính số lượng evaluations cần để đạt fitness ≤ Q3 (cho min problem).
3. Nếu run không bao giờ đạt được → EtTQ = n_eval (10 000).

**EtTQ càng thấp càng tốt**.

---

## 3. Thí nghiệm và Kết quả

### 3.1. Thiết lập chung

**9 EAs**: CMA-ES, DE, PSO, ES-0.02, ES-0.25, ES-0.5, GA-0.02, GA-0.25, GA-0.5.

**57 problems**: 24 (S1) + 9 (S2) + 9 (S3) + 15 (S4).

**n_rep**:
- S1, S2, S3: **30 runs** / (EA, problem).
- S4: **20 runs** (vì tính toán nặng).

**Tổng số runs**: `9 × 42 × 30 + 9 × 15 × 20 = 14 040`.

**Hardware**: 2 máy Ubuntu 20.04, Intel Xeon W-2295 @ 3.0 GHz, 36 cores, 64 GB RAM. Máy không độc quyền cho thí nghiệm → không focus vào execution time trong thảo luận.

**Thời gian / 1 run**:
- S1 synthetic: 1–5 giây.
- S2 regression: ~75 giây (Concrete, Energy), ~250 giây (Wine).
- S3 navigation: 100–200 giây.
- S4 VSR: ~10 000 giây (balance, locomotion), ~2 000 giây (jump).

**Mã nguồn**: framework **JGEA**, public tại GitHub.

---

### 3.2. Kết quả Scenario 1 — Synthetic

**Quan sát tổng thể** (Figure 3, 4 trong paper gốc):
- **CMA-ES là EA hiệu quả và hiệu suất nhất**, thể hiện qua cả phân phối NER lẫn EtTQ.
- NoVS của CMA-ES = **18/24** (xem Table 1), thắng áp đảo.

**Ảnh hưởng của p** (NER vs p, Figure 4):
- Dimensionality **ít ảnh hưởng** đến ranking EAs, trừ PSO và nhẹ hơn là CMA-ES.
- Tác giả giả thuyết **ruggedness quan trọng hơn p** trong việc quyết định hiệu năng.

**Vấn đề với CMA-ES trên rugged landscape**:
- Trên Ackley và Rastrigin (rugged), hiệu năng CMA-ES **giảm** khi p tăng.
- **GA-0.02** tốt hơn CMA-ES trong các trường hợp này.
- NoVS của GA-0.02 = **6**, tương ứng với: 4 bài Rastrigin + Ackley-500 + Sphere-500.

**Phân tích GA theo σ**:
- Với bài toán có optimum tại 0 (Sphere, Ackley, Rastrigin, CPA): **σ nhỏ → tốt hơn**.
- Với PA-1-p, PA-10-p (optimum xa origin): **σ lớn → tốt hơn**.
- Xu hướng này cũng đúng cho ES nhưng rõ rệt hơn ở GA.

---

### 3.3. Kết quả Scenario 2 — ANN-based regression

**Quan sát tổng thể** (Figure 5, 6):
- **CMA-ES nằm trong nhóm tệ nhất**, NoVS = **0**.
- **DE đạt NoVS cao nhất = 7/9**, áp đảo trên regression.

**Ảnh hưởng của p** (Figure 6):
- Khi p tăng, GA-0.02, PSO, ES-0.02 có NER **cao hơn (tệ hơn)** so với DE.
- **GA-0.02** đạt NoVS = 4 trên: Concrete-3, Energy-3, Wine-3, Wine-2 — chủ yếu ở ANN có 3×m hidden neurons.
- **PSO** NoVS = 1 (Concrete-3).
- **ES-0.02** NoVS = 3, đều trên Concrete-ρₕ.

**Efficiency**:
- **GA-0.02 và ES-0.02 có EtTQ tốt nhất**.
- Nhóm {GA-0.02, GA-0.25, ES-0.02, ES-0.25, DE, PSO} có phân phối EtTQ **hẹp hơn nhiều** (consistent) so với nhóm còn lại.

---

### 3.4. Kết quả Scenario 3 — ANN-based 2D navigation

**Quan sát tổng thể** (Figure 7, 8):
- Chênh lệch giữa các EA **ít rõ ràng** hơn S1, S2.
- Problem càng khó (Small → Large → Maze), chênh lệch càng nhỏ, **trừ ES-0.02** — luôn tệ và không ổn định.
- **DE vẫn là EA tốt nhất**, theo sau là GA và ES với σ trung bình/lớn.

**NoVS**:
- DE: **9/9** (thắng toàn bộ).
- GA-0.5: 5.
- GA-0.25: 4.

**Ảnh hưởng của p**: tương tự effectiveness — chênh lệch ít hơn các scenario trước.

---

### 3.5. Kết quả Scenario 4 — ANN-based VSR control

**Lưu ý**: scenario này có cả minimize (3 balance) và maximize (12 bài còn lại).

**Quan sát tổng thể** (Figure 9, 10):
- **DE và GA-0.5** có kết quả tốt nhất về effectiveness.
- Về efficiency, **DE và CMA-ES vượt trội** so với GA-0.5.

**Ảnh hưởng của p** (Figure 10):
- Tương tự S3, p có ít tác động, nhưng NER **không constant** như S3.
- Có **discontinuity nổi bật** từ p=275 (Tower + C) sang p=280 (Biped + C). Khác biệt này không phải do morphology mà do **task** (Tower dùng cho Balance, Biped cho 4 tasks còn lại).

**NoVS**:
- **DE**: 13/15 — chỉ thua trên Balance-C và Flat-C.
- **GA-0.5**: 12 (thua: Balance-C, Balance-HoD, Flat-HoD).
- **GA-0.25**: 11 (thua: Balance-HeD, Jump-HeD, Jump-HoD, Steppy-HeD).
- Với **mỗi** problem VSR, ít nhất 1 trong {DE, GA-0.5, GA-0.25} đạt best hoặc non-worse-than-best.

**CMA-ES trong S4**:
- Tốt hơn hẳn so với S2, S3. NoVS = **5**.
- Best median trên 1 problem: **Jump-HeD**.
- Thắng trên: Flat-HeD, Flat-C, Hilly-C, Jump-HeD, Jump-C.
- **Gặp khó khăn với HoD controller**. Tác giả giả thuyết: HoD (cùng ANN trên mọi voxel) tạo ra dynamics phức tạp → fitness landscape đặc biệt gồ ghề.

---

### 3.6. Tổng hợp NoVS (Table 1 trong paper gốc)

| EA | Synthetic | Regression | 2D nav. | VSR |
|---|---|---|---|---|
| **CMA-ES** | **18** | 0 | 0 | 5 |
| **DE** | 0 | **7** | **9** | **13** |
| **PSO** | 0 | 1 | 2 | 2 |
| **ES-0.02** | 2 | 3 | 0 | 0 |
| **ES-0.25** | 0 | 0 | 3 | 1 |
| **ES-0.5** | 0 | 0 | 3 | 3 |
| **GA-0.02** | 6 | 4 | 0 | 1 |
| **GA-0.25** | 0 | 0 | 4 | 11 |
| **GA-0.5** | 0 | 0 | 5 | **12** |

(Tổng problems mỗi scenario: 24, 9, 9, 15. Đậm = thắng lớn trong scenario đó.)

---

### 3.7. Thảo luận (Discussion)

**Phát hiện chính #1**:
Trên 3 scenarios neuroevolution (S2, S3, S4), **DE luôn đạt NoVS lớn nhất**, theo sau là các biến thể đơn giản của GA hoặc ES.

**Phát hiện chính #2**:
Hiệu năng của một EA trên synthetic problems **không dự đoán được** hiệu năng của nó trên neuroevolution phức tạp hơn.

**Phát hiện chính #3**:
Có mối tương quan giữa **complexity (độ gián tiếp của mapping θ → fitness)** và hiệu năng GA: GA tốt hơn trong các scenarios phức tạp hơn. Đặc biệt GA-0.5 đạt NoVS = 12 trên S4, chỉ thua DE 1 chiến thắng.

**Giả thuyết về landscape ruggedness**:
- Tác giả giả thuyết: complexity của control problem (mức gián tiếp của mapping solution-fitness) **tác động lên ruggedness** của fitness landscape.
- Trên landscape rất gồ ghề, GA với **step-size lớn** khai thác khả năng "nhảy nhanh" giữa các vùng của search space để tìm lời giải tốt.

**Giả thuyết về crossover**:
- 2 EA tốt nhất trên control problems (DE và GA) là **2 EA duy nhất dùng crossover**.
- Crossover tạo lời giải mới bằng cách **recombine các lời giải xa nhau trong search space** — hoạt động như một "bước đi rất lớn".
- Cơ chế này **đặc biệt có lợi** trên rugged landscape của control problems.
- Việc crossover hỗ trợ search trên rugged landscape **không phải quan sát mới** — đã có trong literature (Kolarov 1997, Naumov 2023).

---

## 4. Kết luận (Concluding Remarks)

### 4.1. Những gì đã làm

- So sánh hành vi của nhiều EA khi dùng trong neuroevolution cho continuous control.
- 9 EAs × (24 + 9 + 9 + 15) = 57 problems, chia thành 4 scenarios.
- 3 scenarios là neuroevolution, 2 trong đó là control tasks.
- Scenarios có độ phức tạp tăng dần. Chênh lệch giữa 2 control scenarios đến từ **độ giàu động lực học** của các robotic agents mô phỏng.

### 4.2. Kết quả thực nghiệm

- Các EA đơn giản như **DE và GA đạt effectiveness và efficiency tốt**, đặc biệt trên các problems phức tạp hơn.
- Giả thuyết: hiệu quả này đến từ **khả năng khám phá được tăng cường** và đặc biệt là **crossover**.

### 4.3. Hướng nghiên cứu tương lai

Tác giả dự định điều tra thêm:

1. **Fitness landscape vs problem characteristics**: chiều observation-space, action-space, randomness của môi trường, presence/absence of state trong controller, deceptiveness của goal.
2. **Cá nhân vs xã hội trong học tập**: liệu tương phản giữa **individual learning** (như imitation learning) và **social learning** có tương đồng với tương phản giữa EA không có và có crossover?

---

## 5. Lời cảm ơn

Nghiên cứu thuộc hoạt động R&D của consortium **iNEST** (Interconnected North-Est Innovation Ecosystem), tài trợ bởi EU **Next-GenerationEU** (PNRR Missione 4 Componente 2, Investimento 1.5 – D.D. 1058 23/06/2022, ECS_00000043).

---

## 6. Tham khảo nổi bật (trong bối cảnh bạn cần đọc thêm)

Không phải danh sách đầy đủ — chỉ các reference trọng yếu cho đồ án:

- **[9] Hansen 2016** — *The CMA evolution strategy: A tutorial*. arXiv:1604.00772. **Bắt buộc đọc** nếu muốn hiểu CMA-ES.
- **[41] Storn & Price 1997** — *Differential evolution*. Journal of Global Optimization 11. Paper gốc về DE.
- **[5] Clerc 2012** — *Beyond standard particle swarm optimisation*. Phiên bản PSO mà paper dùng.
- **[7] Derrac et al. 2011** — *Nonparametric statistical tests for comparing EAs*. Swarm and Evolutionary Computation. Cơ sở cho việc dùng Wilcoxon.
- **[45] White et al. 2013** — *Better GP benchmarks*. Genetic Programming and Evolvable Machines. Cơ sở chọn dataset cho S2.
- **[23] Medvet et al. 2020** — *2D-VSR-Sim*. Simulator cho S4.
- **[26] Medvet, Nadizar, Manzoni 2022** — *JGEA*. Framework mà paper dùng.
- **[19] Kolarov 1997** — *Landscape ruggedness in EAs*. IEEE ICEC. Nền tảng lý thuyết cho giả thuyết crossover–ruggedness.
- **[34] Naumov 2023** — *Crossover-based EAs on rugged landscape*. GECCO Companion.

---

## Phụ lục A — Bảng tổng hợp tham số nhanh

| EA | n_pop | Tham số khác | \( \sigma \) variants |
|---|---|---|---|
| **CMA-ES** | default (theo Hansen) | default | — |
| **DE** (DE/rand/1) | 15 | F = 0.5, CR = 0.8 | — |
| **PSO** | 100 | w = 0.8, φ_p = φ_g = 1.5 | — |
| **ES** | 30 | n_parents = 10 | {0.02, 0.25, 0.5} |
| **GA** | 100 | p_xo = 0.8, n_tour = 5 | {0.02, 0.25, 0.5} |

**Chung cho mọi EA**:
- Init: \( \mathbf{x} \sim \mathcal{U}([-1, 1])^p \).
- Termination: `evals > 10 000`.
- Wilcoxon signed-rank, \( \alpha = 0.01 \) cho NoVS.

---

## Phụ lục B — Ký hiệu đã dùng

| Ký hiệu | Ý nghĩa |
|---|---|
| \(p\) | Kích thước search space (số chiều) |
| \( \mathbf{x}, \mathbf{x}^\* \) | Điểm trong \( \mathbb{R}^p \), lời giải tốt nhất |
| \( \boldsymbol{\theta}, \boldsymbol{\theta}^\* \) | Vector tham số ANN (khi bài toán là neuroevolution) |
| \( f(\mathbf{x}), f(\boldsymbol{\theta}) \) | Hàm fitness |
| \( n_{\text{pop}} \) | Kích thước quần thể |
| \( n_{\text{parents}} \) | Số parents tốt nhất trong ES |
| \( n_{\text{tour}} \) | Kích thước tournament trong GA |
| \( p_{\text{xo}} \) | Xác suất crossover |
| \( \sigma \) | Step-size của Gaussian mutation |
| \( \alpha \sim \mathcal{U}([0,1]) \) | Hệ số ngẫu nhiên trong segment geometric crossover |
| \( \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma\mathbf{I}) \) | Nhiễu Gaussian zero-mean |
| \(F\) | Differential weight trong DE |
| \(CR\) | Crossover rate trong DE |
| \(w\) | Inertia trong PSO |
| \( \phi_{\text{particle}}, \phi_{\text{global}} \) | Acceleration coefficients trong PSO |
| \(m\) | Số feature input / số proximity sensor |
| \( \rho_h \) | Relative size của hidden layer (S2) |
| \( n_{\text{rep}} \) | Số run / (EA, problem) |
| \( n_{\text{eval}} \) | Budget evaluations / run (= 10 000) |
| \( \mathrm{NER} \) | Normalized Effectiveness Rank \(\in [0,1]\) |
| \( \mathrm{EtTQ} \) | Evals to Third Quartile |
| \( \mathrm{NoVS} \) | Number of Victories Score |
| \( Q_3 \) / \( f_{75} \) | Quartile thứ 3 của phân phối final fitness |

---

## Ghi chú cuối cho đồ án

Tài liệu này là **tóm tắt–diễn giải** bằng tiếng Việt phục vụ học tập. Khi viết báo cáo hoặc trích dẫn trong đồ án:

1. **Luôn trích dẫn** bản gốc: `El Saliby et al. (2024), GECCO '24 Companion, DOI: 10.1145/3638530.3664112`.
2. **Không copy nguyên đoạn** từ paper gốc vào báo cáo — viết lại bằng lời bạn.
3. **Công thức và số liệu** thì giữ nguyên, vì chúng là facts (không có bản quyền trên facts).
4. **Bảng NoVS** ở mục 3.6 có thể reproduce trong báo cáo với caption "Replicated from Table 1 of El Saliby et al. (2024)".
5. Khi có nghi ngờ về hiểu ngữ nghĩa, luôn quay lại **bản gốc tiếng Anh** làm chuẩn.
