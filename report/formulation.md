# Formulation (Đặc tả bài toán) — Simple Evolutionary Algorithm (Neuroevolution)

Tài liệu này viết phần **formulation** (mô hình hoá bài toán) cho đồ án “Simple Evolutionary Algorithm — Neuroevolution of Continuous Control Policies”. Mục tiêu là mô tả **chính xác**:

- bài toán tối ưu là gì (biến quyết định, không gian tìm kiếm, hàm mục tiêu, ràng buộc),
- cách “một vector tham số” \(\boldsymbol{\theta}\in\mathbb{R}^p\) được diễn giải thành **một chính sách điều khiển/ANN**,
- cách tính **fitness** cho từng scenario (S1–S4),
- ngân sách đánh giá và các **chỉ số** dùng để so sánh thuật toán.

Nội dung bám sát thiết lập chung trong repo (xem `README.md`, `tai_lieu/scenarios_vi.md`, `tai_lieu/algorithms.md`).

---

## 1) Bài toán tối ưu tổng quát

### 1.1. Biến quyết định và không gian tìm kiếm

- **Biến quyết định**: một vector thực
  \[
  \mathbf{x}\in\mathbb{R}^p
  \quad\text{hoặc}\quad
  \boldsymbol{\theta}\in\mathbb{R}^p.
  \]
- **Ý nghĩa**:
  - Scenario 1: \(\mathbf{x}\) **là nghiệm trực tiếp** của bài toán số học.
  - Scenario 2–4: \(\boldsymbol{\theta}\) **là genotype** (vector tham số) của một mạng nơ-ron (ANN). Khi “giải mã” \(\boldsymbol{\theta}\) ta thu được **policy** \(g_{\boldsymbol{\theta}}\) để dự đoán hoặc điều khiển trong mô phỏng.

### 1.2. Khởi tạo (common settings)

Tất cả EA trong đồ án khởi tạo cá thể theo phân phối đều:

\[
\mathbf{x}\sim \mathcal{U}([-1,1]^p).
\]

Điều này quan trọng vì nó làm **công bằng** điểm xuất phát giữa các thuật toán.

### 1.3. Ngân sách đánh giá (termination)

Gọi một lần tính fitness là một **fitness evaluation**.

- Mỗi run dừng ở iteration đầu tiên mà tổng số fitness evaluations **vượt**:
  \[
  n_{\text{eval}} = 10\,000.
  \]

Vì fitness ở S3/S4 đến từ mô phỏng (tốn thời gian), việc “tính theo số lần gọi fitness” là cách chuẩn để so sánh công bằng.

---

## 2) “Pipeline” chung của một thuật toán tiến hoá (EA)

Một EA trong đồ án có thể được nhìn như tối ưu hoá hộp đen:

\[
\min_{\mathbf{x}\in\mathbb{R}^p} f(\mathbf{x})
\quad\text{hoặc}\quad
\min_{\boldsymbol{\theta}\in\mathbb{R}^p} f(\boldsymbol{\theta}).
\]

Trong đó \(f(\cdot)\) là **hàm fitness**:

- S1: tính trực tiếp từ công thức benchmark.
- S2: chạy ANN trên dataset → tính MSE.
- S3/S4: chạy mô phỏng điều khiển → đo hiệu năng (khoảng cách, vận tốc, độ cao, góc, …).

### 2.1. Các thành phần tối thiểu

Với một EA bất kỳ, ta thường có:

- **Population** \(P_t=\{\mathbf{x}^{(1)}_t,\dots,\mathbf{x}^{(n_{\text{pop}})}_t\}\).
- **Evaluation**: tính \(f(\mathbf{x})\) cho mọi cá thể.
- **Selection**: chọn cá thể tốt để sinh thế hệ sau.
- **Variation**: tạo cá thể mới bằng *mutation* và/hoặc *crossover* (tuỳ EA).
- **Replacement/Elitism**: tạo quần thể mới, thường giữ lại nghiệm tốt nhất để ổn định.

### 2.2. Minimize vs maximize (chuẩn hoá mục tiêu)

Trong repo:

- S1, S2, S3 chủ yếu là **minimize**.
- S4 có cả **maximize** (locomotion/jump) và **minimize** (balance).

Để thống nhất, có 2 cách ghi formulation:

1) Viết đúng mục tiêu từng bài:
\[
\min f(\cdot)\quad \text{hoặc}\quad \max f(\cdot).
\]

2) Quy về minimize bằng cách đổi dấu:
\[
\min \tilde f(\cdot),\quad \tilde f = 
\begin{cases}
f & \text{nếu bài là minimize}\\
-f & \text{nếu bài là maximize}
\end{cases}
\]

Trong phần dưới, mình sẽ ghi đúng theo mục tiêu gốc từng scenario và chỉ ra khi nào cần “đổi dấu” để code thuận tiện.

---

## 3) Formulation theo từng scenario

### Ký hiệu chung cho S2–S4 (neuroevolution)

- \(g_{\boldsymbol{\theta}}\): ANN/policy được tham số hoá bởi \(\boldsymbol{\theta}\).
- \(\text{decode}(\boldsymbol{\theta}) \mapsto\) trọng số/bias của các lớp ANN.
- Fitness là một hàm hợp:
\[
f(\boldsymbol{\theta}) = \Phi\big(g_{\boldsymbol{\theta}}\big)
\]
với \(\Phi\) là “hàm đánh giá” (MSE hoặc mô phỏng).

Điểm làm neuroevolution khó hơn tối ưu số học là mapping \(\boldsymbol{\theta}\mapsto f(\boldsymbol{\theta})\) **gián tiếp** và thường **gồ ghề/stochastic** (đặc biệt ở S3 do khởi tạo ngẫu nhiên và ở S4 do mô phỏng phức tạp).

---

## 3.1) Scenario 1 — Synthetic numerical benchmarks

### (a) Bài toán

Với mỗi problem, ta giải:

\[
\min_{\mathbf{x}\in\mathbb{R}^p} f(\mathbf{x}).
\]

Trong paper và repo, \(p\in\{20,100,200,500\}\).

### (b) Hàm mục tiêu điển hình

1) **Sphere**:
\[
f(\mathbf{x})=\sum_{i=1}^p x_i^2,\quad \mathbf{x}^\star=\mathbf{0}.
\]

2) **Point Aiming (PA)** (optimum không ở gốc):
\[
f(\mathbf{x})=\|\mathbf{x}-\mathbf{x}^\star\|_2,
\quad \mathbf{x}^\star=\mathbf{1}\ \text{hoặc}\ 10\mathbf{1}.
\]

3) **Circular Point Aiming (CPA)**:
\[
f(\mathbf{x})=\min_{i\in\{1,\dots,5\}} \|\mathbf{x}-\mathbf{x}_i^\star\|_2.
\]

4) **Ackley**, 5) **Rastrigin**: các benchmark multimodal (định nghĩa chuẩn).

### (c) Ví dụ minh hoạ (Sphere 2D, 1 bước đánh giá)

Giả sử \(p=2\), \(\mathbf{x}=(0.6,-0.8)\). Khi đó:

\[
f(\mathbf{x}) = 0.6^2 + (-0.8)^2 = 0.36 + 0.64 = 1.0.
\]

Nếu một EA tạo ra \(\mathbf{x}'=(0.1,-0.2)\) thì:
\[
f(\mathbf{x}') = 0.01 + 0.04 = 0.05
\]
⇒ \(\mathbf{x}'\) tốt hơn rõ rệt (vì mục tiêu là minimize).

---

## 3.2) Scenario 2 — ANN-based Regression (tối ưu MSE)

### (a) Dữ liệu và mục tiêu

Cho dataset:
\[
D = \left\{(\mathbf{x}^{(i)}, \tilde y^{(i)})\right\}_{i=1}^{n},
\quad \mathbf{x}^{(i)}\in\mathbb{R}^m,\ \tilde y^{(i)}\in\mathbb{R}.
\]

Ta tối ưu tham số ANN \(\boldsymbol{\theta}\) để minimize:

\[
f(\boldsymbol{\theta}) = \mathrm{MSE}(g_{\boldsymbol{\theta}}, D)
=\frac{1}{n}\sum_{i=1}^{n}\left(g_{\boldsymbol{\theta}}(\mathbf{x}^{(i)})-\tilde y^{(i)}\right)^2.
\]

Tiền xử lý theo paper/repo:

- Chuẩn hoá input \(\mathbf{x}\),
- Rescale nhãn \(\tilde y\) về \([-1,1]\) để tương thích `tanh`.

### (b) Kiến trúc ANN và số chiều \(p\)

ANN feed-forward, 1 hidden layer, `tanh`.

- Input: \(m\)
- Hidden: \(\rho_h m\) với \(\rho_h\in\{1,2,3\}\)
- Output: 1

Số tham số (bao gồm cả bias):

\[
p = (m+1)\rho_h m + (\rho_h m + 1).
\]

Diễn giải:

- Từ input→hidden: ma trận trọng số kích thước \((\rho_h m)\times m\) và bias \((\rho_h m)\) ⇒ tổng \((m+1)\rho_h m\).
- Từ hidden→output: trọng số kích thước \(1\times(\rho_h m)\) và bias 1 ⇒ \(\rho_h m+1\).

### (c) Ví dụ minh hoạ (Concrete/Energy: \(m=8\), \(\rho_h=2\))

- Hidden size: \(\rho_h m = 16\).
- Số tham số:
\[
p=(8+1)\cdot 16 + (16+1)=144+17=161.
\]

Khi đó:
- genotype là \(\boldsymbol{\theta}\in\mathbb{R}^{161}\),
- một lần **fitness evaluation** là:
  1) `decode(θ)` thành \(W_1,b_1,W_2,b_2\),
  2) chạy forward qua toàn bộ dataset (hoặc tập đánh giá cố định theo repo),
  3) tính MSE theo công thức trên.

---

## 3.3) Scenario 3 — ANN-based 2D Navigation (điều khiển robot)

### (a) Trạng thái cảm biến và policy

Robot differential-drive, policy là ANN:

\[
g_{\boldsymbol{\theta}}:\mathbb{R}^{m+2}\to\mathbb{R}^{2}.
\]

Trong đó:

- \(m\) proximity sensors (phủ nửa cung trước robot),
- 2 cảm biến liên quan tới target (khoảng cách và góc),
- output là 2 giá trị điều khiển (vận tốc bánh trái/phải hoặc tương đương).

### (b) Môi trường và tính stochastic

Mô phỏng trong arena \(1m\times 1m\), target cố định tại \((0.50,0.15)\), thời gian mô phỏng 60s với \(\Delta t=0.1s\).

Trạng thái ban đầu ngẫu nhiên:

- \(x_0\sim U([0.45,0.55])\),
- \(y_0\sim U([0.80,0.85])\).

Do khởi tạo ngẫu nhiên, cùng một \(\boldsymbol{\theta}\) có thể cho fitness khác nhau theo seed ⇒ landscape có nhiễu (stochastic objective).

### (c) Fitness (minimize)

Theo paper tóm tắt trong `tai_lieu/scenarios_vi.md`, fitness được định nghĩa là **khoảng cách trung bình** từ robot tới target trong suốt simulation:

\[
f(\boldsymbol{\theta}) = \frac{1}{T}\sum_{t=1}^{T} d\big(\mathbf{p}_t, \mathbf{p}_{\text{target}}\big),
\]

với \(\mathbf{p}_t\) là vị trí robot ở bước \(t\), \(T\) là tổng số bước mô phỏng, và \(d(\cdot,\cdot)\) là khoảng cách Euclid.

### (d) Ví dụ minh hoạ (1 rollout rút gọn)

Giả sử mô phỏng rút gọn chỉ có 5 bước (để minh hoạ cách “tính trung bình”), và khoảng cách đến target đo được:

\[
d_1=0.70,\ d_2=0.55,\ d_3=0.40,\ d_4=0.35,\ d_5=0.30.
\]

Khi đó:

\[
f(\boldsymbol{\theta})=\frac{0.70+0.55+0.40+0.35+0.30}{5} = 0.46.
\]

Nếu một \(\boldsymbol{\theta}'\) khác cho dãy khoảng cách \([0.80,0.75,0.70,0.70,0.68]\) thì fitness trung bình \(=0.726\) ⇒ kém hơn.

---

## 3.4) Scenario 4 — VSR (Voxel-based Soft Robot) Control

### (a) Đối tượng tối ưu

Tối ưu tham số ANN controller \(\boldsymbol{\theta}\) để điều khiển robot mềm (morphology cố định: **Biped** hoặc **Tower**).

Tuỳ kiểu controller, cách “đóng gói” \(\boldsymbol{\theta}\) khác nhau:

- **C (centralized)**: 1 ANN đọc toàn bộ sensor readings, xuất action cho các voxel.
- **HoD (homo-distributed)**: mỗi voxel có 1 ANN nhưng **weight-sharing** (cùng \(\boldsymbol{\theta}\)).
- **HeD (hetero-distributed)**: mỗi voxel có 1 ANN **tham số riêng**; \(\boldsymbol{\theta}\) là phép nối (concat) tất cả tham số.

### (b) Sensor và nhiễu

Sensor readings được normalize về \([-1,1]\) và được cộng Gaussian noise (variance 0.05). Controller chỉ được gọi mỗi 0.2s (giữ output giữa các lần gọi).

### (c) Fitness (min/max)

S4 gồm nhiều task, trong đó:

- **Locomotion** (flat/hilly/steppy): **maximize** vận tốc trung bình theo trục \(x\) của COM trong 30s.
- **Jump**: **maximize** độ cao cực đại của COM trong 10s (bỏ 5s đầu).
- **Balance**: **minimize** góc trung bình của swing trong 30s, có phạt khi chạm đất.

Tuỳ task, formulation là:

\[
\max_{\boldsymbol{\theta}\in\mathbb{R}^p} f(\boldsymbol{\theta})
\quad\text{hoặc}\quad
\min_{\boldsymbol{\theta}\in\mathbb{R}^p} f(\boldsymbol{\theta}).
\]

Khi triển khai EA dạng “minimize”, thường dùng \(\tilde f=-f\) cho các task maximize.

---

## 4) Chỉ số đánh giá trong báo cáo (metrics)

Phần này không phải “formulation của bài toán con”, nhưng là formulation của **bài toán so sánh thuật toán** trong báo cáo.

### 4.1. Final best fitness

Với mỗi run (một EA trên một problem với một seed), giá trị báo cáo cơ bản nhất là:

- lấy cá thể tốt nhất ở thời điểm dừng (budget \(10\,000\) evals),
- ghi nhận fitness cuối cùng (“final best fitness”).

### 4.2. NER (Normalized Effectiveness Rank)

NER chuẩn hoá thứ hạng final best fitness giữa các EA trong cùng một problem rồi gộp theo scenario, cho kết quả \(\mathrm{NER}\in[0,1]\) với **lower = better**. NER cho phép so sánh **tương đối** khi các problem có thang đo fitness khác nhau.

### 4.3. EtTQ (Evals to Third Quartile)

EtTQ là số fitness evaluations cần để một run đạt mức “đủ tốt” (không tệ hơn ngưỡng quartile 75% của phân phối final fitness). **Lower = better**; nếu không đạt thì EtTQ = \(10\,000\).

### 4.4. NoVS (Number of Victories Score)

NoVS là số problem mà một EA được tính “thắng” theo kiểm định Wilcoxon signed-rank so với EA có median tốt nhất ở problem đó (mức ý nghĩa \(\alpha=0.01\)).

---

## 5) Gợi ý cách đưa formulation này vào báo cáo chính

Nếu bạn đang viết `BAO_CAO_DO_AN.md`/báo cáo tổng, bạn có thể:

- dùng **Mục 1–2** làm “Định nghĩa bài toán & thiết lập chung”,
- dùng **Mục 3** làm “Mô tả 4 kịch bản thí nghiệm” (kèm công thức fitness),
- dùng **Mục 4** làm “Chỉ số đánh giá và phương pháp thống kê”.

