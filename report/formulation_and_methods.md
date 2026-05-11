# Tổng hợp chi tiết: Formulation + Methods

Tài liệu này là bản **tổng hợp đầy đủ** hai phần:

- `report/formulation.md` (mô hình hoá bài toán/fitness)
- `report/method.md` (phương pháp/thiết lập thực nghiệm/triển khai)

Mục tiêu là tạo một chương “đọc một lần là nắm toàn bộ pipeline” của đồ án: **từ bài toán tối ưu** → **cách ánh xạ genotype \(\boldsymbol{\theta}\)** → **cách tính fitness trong 4 scenario** → **9 thuật toán tiến hoá** → **quy trình chạy thí nghiệm + log + metrics**.

Nội dung bám sát paper El Saliby et al. (2024) và đúng với code trong repo (`Scenario_1_run.py` … `Scenario_4_run.py`), đồng thời tương thích với tài liệu `tai_lieu/scenarios_vi.md` và `tai_lieu/algorithms.md`.

---

## 0) Ký hiệu & quy ước chung

### 0.1. Bài toán tối ưu

Ta xét bài toán tối ưu dạng hộp đen trong không gian thực:

\[
\mathbf{z}\in \mathbb{R}^p,\qquad
\min_{\mathbf{z}} f(\mathbf{z})
\quad\text{hoặc}\quad
\max_{\mathbf{z}} f(\mathbf{z}).
\]

Trong đồ án, \(\mathbf{z}\) có 2 cách hiểu:

- \(\mathbf{z}=\mathbf{x}\): nghiệm trực tiếp (Scenario 1).
- \(\mathbf{z}=\boldsymbol{\theta}\): vector tham số ANN (Scenario 2–4).

### 0.2. Fitness evaluation và ngân sách công bằng

- **1 fitness evaluation** = 1 lần gọi \(f(\cdot)\).
- Mỗi run dừng ở iteration đầu tiên mà tổng số evaluation **vượt**:

\[
n_{\text{eval}} = 10\,000.
\]

Đây là tiêu chuẩn “fair budget” (đặc biệt quan trọng khi S3/S4 phải chạy mô phỏng tốn thời gian).

### 0.3. Khởi tạo thống nhất cho mọi thuật toán

Tất cả EA (kể cả CMA-ES) khởi tạo từ phân phối đều:

\[
\mathbf{z}_0 \sim \mathcal{U}([-1,1]^p).
\]

Với các EA theo quần thể, điều này tương đương với khởi tạo từng cá thể trong population theo cùng phân phối.

### 0.4. Quy về bài toán minimize (nếu cần)

Trong Scenario 4 có bài toán maximize. Để dùng chung các solver hướng minimize, ta dùng:

\[
\tilde f(\mathbf{z})=
\begin{cases}
f(\mathbf{z}) & \text{nếu mục tiêu gốc là minimize}\\
-f(\mathbf{z}) & \text{nếu mục tiêu gốc là maximize}
\end{cases}
\]

và luôn tối ưu \(\min \tilde f(\mathbf{z})\).

---

## 1) Formulation tổng quát cho neuroevolution (S2–S4)

### 1.1. Genotype → Phenotype (giải mã ANN)

Trong neuroevolution, cá thể là một vector thực:

\[
\boldsymbol{\theta}\in\mathbb{R}^p.
\]

Ta có một hàm “giải mã”:

\[
\text{decode}(\boldsymbol{\theta})\longrightarrow \{W_\ell, b_\ell\}_{\ell}
\]

để biến \(\boldsymbol{\theta}\) thành trọng số/bias của ANN. ANN đó định nghĩa policy:

\[
g_{\boldsymbol{\theta}}(\cdot)\quad\text{(dự đoán hoặc điều khiển)}.
\]

Khi đó, fitness là hàm hợp:

\[
f(\boldsymbol{\theta}) = \Phi(g_{\boldsymbol{\theta}}),
\]

với \(\Phi\) là thủ tục đánh giá (MSE trên dataset, hoặc mô phỏng episode).

### 1.2. Điểm khác biệt giữa S1 và S2–S4

- **S1 (synthetic)**: \(f(\mathbf{x})\) tính trực tiếp từ công thức ⇒ deterministic, tương đối “mịn”.
- **S2 (regression)**: \(f(\theta)\) là MSE ⇒ deterministic, nhưng mapping \(\theta\mapsto f\) vẫn phức tạp.
- **S3 (navigation)**: fitness stochastic do khởi tạo ngẫu nhiên mỗi episode.
- **S4 (VSR)**: fitness dựa trên mô phỏng (và có noise sensor) ⇒ landscape gồ ghề, gián tiếp.

Điểm mấu chốt của đồ án/paper: **hiệu năng trên S1 không dự đoán tốt S2–S4**; các EA có crossover (DE/GA) thường lợi thế trên neuroevolution phức tạp.

---

## 2) Formulation theo từng Scenario (S1–S4) + ví dụ minh hoạ

## 2.1) Scenario 1 — Synthetic numerical benchmarks (24 problems)

### (a) Bài toán

Với mỗi dimension \(p\in\{20,100,200,500\}\), giải:

\[
\min_{\mathbf{x}\in\mathbb{R}^p} f(\mathbf{x}).
\]

### (b) Các hàm benchmark (tóm tắt)

- **Sphere**:
  \[
  f(\mathbf{x})=\sum_{i=1}^{p} x_i^2.
  \]
- **Point Aiming (PA)**: \(f(\mathbf{x})=\|\mathbf{x}-\mathbf{x}^\star\|_2\) với \(\mathbf{x}^\star=\mathbf{1}\) hoặc \(10\mathbf{1}\).
- **Circular Point Aiming (CPA)**:
  \[
  f(\mathbf{x})=\min_{i=1..5}\|\mathbf{x}-\mathbf{x}_i^\star\|_2,
  \]
  với 5 điểm \(\mathbf{x}_i^\star\) nằm trên đường tròn (bán kính 2, tâm \(\mathbf{1}\)) trong mặt phẳng \((x_1,x_2)\), các chiều còn lại cố định bằng 1.
- **Ackley**, **Rastrigin**: multimodal, rugged.

### (c) Ví dụ minh hoạ (Sphere 2D)

Với \(\mathbf{x}=(0.6,-0.8)\):

\[
f(\mathbf{x})=0.6^2+(-0.8)^2=1.0.
\]

Nếu EA sinh \(\mathbf{x}'=(0.1,-0.2)\):

\[
f(\mathbf{x}')=0.05
\]

⇒ \(\mathbf{x}'\) tốt hơn (vì minimize).

---

## 2.2) Scenario 2 — ANN-based Regression (9 problems)

### (a) Bài toán

Cho dataset:

\[
D=\{(\mathbf{x}^{(i)}, \tilde y^{(i)})\}_{i=1}^{n},
\quad \mathbf{x}^{(i)}\in\mathbb{R}^m.
\]

Tối ưu tham số ANN \(\boldsymbol{\theta}\) để minimize:

\[
f(\boldsymbol{\theta})
=\frac{1}{n}\sum_{i=1}^{n}\left(g_{\boldsymbol{\theta}}(\mathbf{x}^{(i)})-\tilde y^{(i)}\right)^2.
\]

### (b) Kiến trúc ANN

- 1 hidden layer, `tanh`
- hidden size \(h=\rho_h m\), \(\rho_h\in\{1,2,3\}\)
- output = 1

Số tham số:

\[
p=(m+1)h + (h+1) = (m+1)\rho_h m + (\rho_h m+1).
\]

### (c) Cách tính fitness trong code (Methods-level)

Một evaluation gồm:

1) chuẩn hoá \(X\), rescale \(y\rightarrow[-1,1]\) (đã cache `.npz`),
2) `decode(θ)` thành \(W_1,b_1,W_2,b_2\),
3) forward qua toàn bộ dataset, tính MSE.

### (d) Ví dụ minh hoạ tính \(p\)

Với \(m=8\) và \(\rho_h=2\) ⇒ \(h=16\):

\[
p=(8+1)\cdot 16 + (16+1)=161.
\]

Nghĩa là genotype \(\theta\in\mathbb{R}^{161}\).

---

## 2.3) Scenario 3 — ANN-based 2D Navigation (9 problems)

### (a) Policy và input/output

Robot differential-drive, ANN:

\[
g_{\boldsymbol{\theta}}:\mathbb{R}^{m+2}\to\mathbb{R}^{2}.
\]

- input: \(m\) proximity sensors + (distance-to-target, angle-to-target)
- output: 2 giá trị điều khiển 2 bánh (mỗi bánh \(\in[-V_{\max},V_{\max}]\) sau khi scale từ `tanh`).

### (b) Mô phỏng và stochasticity

- Arena: \(1m\times 1m\) với 3 biến thể: Small / Large / Maze.
- Target cố định \((0.50,0.15)\).
- \(\Delta t=0.1s\), \(T=60s\).
- Init ngẫu nhiên mỗi episode:
  - \(x_0\sim U([0.45,0.55])\),
  - \(y_0\sim U([0.80,0.85])\),
  - heading ngẫu nhiên.

Fitness của cùng \(\theta\) thay đổi theo episode ⇒ stochastic objective.

### (c) Fitness (minimize)

Fitness là khoảng cách trung bình robot–target trong suốt episode:

\[
f(\boldsymbol{\theta})
=\frac{1}{T}\sum_{t=1}^{T} \|\mathbf{p}_t-\mathbf{p}_{\text{target}}\|_2.
\]

### (d) Cách “làm stochastic nhưng tái lập”

Trong code, mỗi lần gọi fitness tăng một `eval_counter` để tạo seed mô phỏng mới theo:

\[
\text{seed}_{\text{sim}}=\text{seed\_offset} + \text{eval\_counter}.
\]

Nhờ vậy:

- cùng EA-seed ⇒ chạy lại vẫn ra đúng chuỗi episode,
- nhưng mỗi evaluation vẫn là “episode mới” (giống paper).

### (e) Ví dụ minh hoạ tính mean distance

Giả sử 5 bước đo được khoảng cách:
\([0.70,0.55,0.40,0.35,0.30]\) ⇒ fitness:

\[
f=\frac{0.70+0.55+0.40+0.35+0.30}{5}=0.46.
\]

---

## 2.4) Scenario 4 — VSR Control (15 problems)

### (a) Mục tiêu

Tối ưu ANN controller cho robot mềm (Biped/Tower). Controller có 3 kiểu:

- **C (centralized)**: 1 ANN đọc toàn bộ sensor, output contraction cho mọi voxel.
- **HoD (homo-distributed)**: mỗi voxel chạy 1 ANN nhưng **chia sẻ tham số**.
- **HeD (hetero-distributed)**: mỗi voxel 1 ANN tham số riêng; genotype là concat.

### (b) Sensor, noise và tần suất điều khiển

- Sensor output normalize về \([-1,1]\).
- Thêm Gaussian noise variance 0.05.
- Controller gọi mỗi 0.2s, physics step 60Hz.

### (c) Tasks và objective

S4 gồm:

- locomotion (flat/hilly/steppy): maximize vận tốc trung bình theo \(x\),
- jump: maximize độ cao cực đại (bỏ 5s đầu),
- balance: minimize dao động/độ lệch (paper dùng góc + malus; code dùng proxy drift).

### (d) Quy về minimize trong code

Nếu metric gốc cần maximize, code trả về \(-\text{metric}\) để solver minimize thống nhất.

---

## 3) Methods — Bộ 9 thuật toán tiến hoá (paper-exact hyperparameters)

### 3.1. Nguyên tắc chung của một solver trong repo

Mỗi solver có cấu trúc:

- init population / init state,
- lặp cho tới khi `total_evals >= n_evals`,
- đánh giá fitness,
- cập nhật theo cơ chế của thuật toán,
- ghi log mỗi 5 iterations bằng `_record(...)`.

Trong CSV, mỗi dòng record gắn thêm metadata:

- `seed` (1-index),
- `problem` (tên problem),
- `solver_sigma` (tên solver + sigma nếu có),
- `objective` (minimize/maximize, hoặc gốc),
- `genotype_size` = \(p\).

### 3.2. CMA-ES

- Khởi tạo: \(x_0\sim U([-1,1]^p)\), \(\sigma_0=0.5\).
- Mỗi vòng:
  - lấy mẫu `ask()` ra \(\lambda\) nghiệm,
  - đánh giá \(f\),
  - `tell()` để cập nhật.
- Tham số còn lại: vanilla defaults của thư viện `cma` (phù hợp paper).

### 3.3. DE/rand/1/bin

Hyperparameters: \(NP=15\), \(F=0.5\), \(CR=0.8\).

Với mỗi target \(\mathbf{x}_i\):

1) chọn \(r_1,r_2,r_3\) khác nhau (\(\neq i\)),
2) mutant: \(\mathbf{v}_i=\mathbf{x}_{r_1}+F(\mathbf{x}_{r_2}-\mathbf{x}_{r_3})\),
3) crossover theo chiều để tạo trial \(\mathbf{u}_i\),
4) nếu \(f(\mathbf{u}_i)\le f(\mathbf{x}_i)\) thì thay.

DE và GA là 2 họ **có recombination/crossover** “đúng nghĩa” trong bộ so sánh — điểm quan trọng cho discussion.

### 3.4. PSO

Hyperparameters: \(n_{\text{pop}}=100\), \(\omega=0.8\), \(\phi_p=\phi_g=1.5\).

Mỗi iteration:

- cập nhật velocity theo personal/global best,
- cập nhật position,
- đánh giá lại toàn bộ swarm,
- cập nhật pbest/gbest.

### 3.5. ES (basic) — ES-0.02 / ES-0.25 / ES-0.5

Hyperparameters:

- \(n_{\text{pop}}=30\),
- \(n_{\text{parents}}=10\),
- elitism giữ best,
- \(\sigma\in\{0.02,0.25,0.5\}\).

Mỗi generation:

1) sort theo fitness,
2) \(\mu\leftarrow\) mean top parents,
3) sinh \(n_{\text{pop}}-1\) điểm mới từ \(\mathcal{N}(\mu,\sigma I)\).

**Chi tiết triển khai**: \(\sigma\) được hiểu là variance ⇒ std = \(\sqrt{\sigma}\) khi sample.

### 3.6. GA số thực — GA-0.02 / GA-0.25 / GA-0.5

Hyperparameters:

- \(n_{\text{pop}}=100\),
- tournament size \(n_{\text{tour}}=5\),
- \(p_{\text{xo}}=0.8\),
- \(\sigma\in\{0.02,0.25,0.5\}\) (variance).

Mỗi generation:

1) sinh `offspring` size \(n_{\text{pop}}\):
   - crossover: \(\mathbf{x}'=\mathbf{x}_1+\alpha(\mathbf{x}_2-\mathbf{x}_1)+\epsilon\),
   - mutation: \(\mathbf{x}'=\mathbf{x}+\epsilon\),
   với \(\epsilon\sim \mathcal{N}(0,\sigma I)\).
2) đánh giá offspring,
3) merge (parent + offspring) rồi giữ top \(n_{\text{pop}}\).

---

## 4) Thiết lập thực nghiệm (Experimental protocol)

### 4.1. Số problem và số run

- S1: 24 problems, \(n_{\text{rep}}=30\)
- S2: 9 problems, \(n_{\text{rep}}=30\)
- S3: 9 problems, \(n_{\text{rep}}=30\)
- S4: 15 problems, \(n_{\text{rep}}=20\) (do mô phỏng nặng)

Tổng số runs (theo paper) rất lớn; repo hỗ trợ `--quick` để smoke test pipeline.

### 4.2. Song song hoá

Các script dùng multiprocessing với start method `spawn` (phù hợp Windows). Mỗi task là một tuple:

- (problem spec, solver, sigma nếu có, seed, n_evals, …).

Riêng S2 có initializer để load dataset theo-process nhằm tránh pickle mảng lớn.

### 4.3. Chế độ `--quick`

Mục đích: chạy nhanh để kiểm tra cài đặt và pipeline xuất file.

Ví dụ:

- S1: giảm dims/probem subset, `n_evals` nhỏ.
- S3/S4: rút ngắn thời gian mô phỏng (episode ngắn hơn).

---

## 5) Logging, dữ liệu đầu ra và cách dùng cho phần Results

### 5.1. Log schema (record)

Mỗi solver ghi record định kỳ với các trường:

- `iterations`, `evals`, `births`, `elapsed`,
- `best_fitness`,
- `geno_uni`, `fit_uni` (ước lượng diversity: unique ratio sau khi làm tròn).

Ý nghĩa:

- `best_fitness` theo thời gian phục vụ plot convergence,
- `evals` làm trục hoành để so sánh theo budget,
- `geno_uni/fit_uni` giúp quan sát “quần thể có bị co cụm sớm không”.

### 5.2. File kết quả CSV

Mỗi scenario xuất CSV (phân tách `;`) tại:

- `Scenario_1/results/Scenario_1_Novel_Final.csv`
- `Scenario_2/results/Scenario_2_Novel_Final.csv`
- `Scenario_3/results/Scenario_3_Novel_Final.csv`
- `Scenario_4/results/Scenario_4_Novel_Final.csv`

Các notebook (`Scenario_*_local.ipynb`) dùng CSV để sinh:

- convergence plot (median + IQR),
- phân phối final best fitness,
- NER, EtTQ, NoVS, và đồ thị NER theo \(p\).

---

## 6) Metrics dùng trong báo cáo (định nghĩa ngắn gọn)

### 6.1. Final best fitness

Với một run: best fitness tại thời điểm dừng theo budget.

### 6.2. NER (Normalized Effectiveness Rank)

Chuẩn hoá thứ hạng final fitness trong từng problem, sau đó gộp trong scenario. NER \(\in[0,1]\), **lower = better**.

### 6.3. EtTQ (Evals to Third Quartile)

Số evaluations cần để một run đạt mức “không tệ hơn” ngưỡng quartile 75% của phân phối final fitness. **Lower = better**; không đạt ⇒ EtTQ = 10,000.

### 6.4. NoVS (Number of Victories Score)

Số problem mà EA “không khác biệt có ý nghĩa” so với EA median-best theo Wilcoxon signed-rank test (\(\alpha=0.01\)).

---

## 7) Gợi ý dàn ý đưa vào báo cáo đồ án

Nếu bạn muốn dùng nguyên chương này cho báo cáo tổng, bố cục gợi ý:

- **Mục 0–1**: Thiết lập chung + formulation neuroevolution.
- **Mục 2**: Formulation theo từng scenario (đây là “bài toán gì?”).
- **Mục 3–4**: Methods (thuật toán nào? chạy thế nào? tham số nào?).
- **Mục 5–6**: Dữ liệu đầu ra + metrics (cầu nối sang Results).

