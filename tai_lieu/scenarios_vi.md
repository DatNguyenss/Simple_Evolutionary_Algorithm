# Bốn scenario trong paper (GECCO ’24 Companion) — trình bày theo đúng thiết lập tác giả

Tài liệu này **chỉ** tóm tắt phần *bài toán + thông số + thống kê/kết quả* theo paper:

- Michel El Saliby, Giorgia Nadizar, Erica Salvato, Eric Medvet — *A Simple EA is All You Need for Neuroevolution of Continuous Control Policies* — **GECCO ’24 Companion** (Melbourne, VIC, Australia).

Mục tiêu của paper là so sánh các EA “đơn giản” trên các bài toán tối ưu hoá số trong \(\mathbb{R}^p\), trong đó **scenario 1** là tối ưu trực tiếp \(f(\mathbf{x})\), còn **scenario 2–4** là neuroevolution: mỗi điểm \(\boldsymbol{\theta}\in\mathbb{R}^p\) là vector tham số của ANN và fitness \(f(\boldsymbol{\theta})\) được định nghĩa theo MSE hoặc theo mô phỏng điều khiển.

---

## 1) Thiết lập chung cho mọi EA (Methods 2.1 + termination)

### 1.1. Không gian tìm kiếm và khởi tạo

- Không gian tìm kiếm: \(\mathbb{R}^p\).
- Khởi tạo quần thể/điểm ban đầu: **uniform** trên \([-1,1]^p\) (áp dụng cho tất cả EA).

### 1.2. Điều kiện dừng (budget công bằng)

- Dừng ở **iteration đầu tiên** mà số lần gọi fitness evaluation **vượt quá** \(n_{\mathrm{eval}}=10\,000\).

### 1.3. Bộ EA được so sánh (9 “EA instances”)

Paper so sánh **9 cấu hình** (một số là cùng họ EA nhưng khác \(\sigma\)):

- **CMA-ES**: “vanilla” theo Hansen (tutorial), **mặc định toàn bộ hyper-parameters**.
- **DE /rand/1**: \(n_{\mathrm{pop}}=NP=15\), \(F=0.5\), \(CR=0.8\).
- **PSO** (phiên bản theo Clerc 2012): \(n_{\mathrm{pop}}=100\), cognitive weight \(w=0.8\), hệ số gia tốc cognitive \(\phi_{\mathrm{particle}}=1.5\), social \(\phi_{\mathrm{global}}=1.5\).
- **ES (basic variant)**: \(n_{\mathrm{pop}}=30\), \(n_{\mathrm{parents}}=\lfloor 0.33\,n_{\mathrm{pop}}\rfloor=10\), cập nhật quần thể bằng Gaussian noise quanh mean của parents với phương sai \(\sigma^2\) theo từng chiều (identity covariance). Thử **3** giá trị \(\sigma\in\{0.02,0.25,0.5\}\) → **ES-0.02 / ES-0.25 / ES-0.5**.
- **GA (standard variant cho continuous)**:
  - \(n_{\mathrm{pop}}=100\), xác suất chọn crossover \(p_{\mathrm{xo}}=0.8\) (còn lại mutation), tournament size \(n_{\mathrm{tour}}=5\).
  - Mutation Gaussian: \(\mathbf{x}'=\mathbf{x}+\boldsymbol{\epsilon}\), \(\boldsymbol{\epsilon}\sim\mathcal{N}(0,\sigma^2 I)\).
  - Crossover “segment geometric”: \(\mathbf{x}'=\mathbf{x}_1+\alpha(\mathbf{x}_2-\mathbf{x}_1)+\boldsymbol{\epsilon}\), \(\alpha\sim U([0,1])\), \(\boldsymbol{\epsilon}\sim\mathcal{N}(0,\sigma^2 I)\).
  - Thử **3** \(\sigma\in\{0.02,0.25,0.5\}\) → **GA-0.02 / GA-0.25 / GA-0.5**.

**Nhận xét theo paper**: chỉ **DE** và **GA** dùng “crossover đúng nghĩa” (recombination) trong bộ thuật toán được chọn.

---

## 2) Scenario 1 (S1): synthetic numerical benchmarks (Section 2.2.1)

### 2.1. Mục tiêu bài toán

Tối thiểu hoá \(f(\mathbf{x})\) với \(\mathbf{x}\in\mathbb{R}^p\). Paper chọn 5 họ hàm benchmark (mỗi họ nhằm kiểm tra một “tính chất landscape” khác nhau: unimodal/multimodal, regular/irregular, separable/non-separable, và ảnh hưởng của **khoảng cách optimum tới gốc toạ độ**).

### 2.2. Danh sách hàm + định nghĩa chính (theo paper)

1. **Sphere** (unimodal, dễ nhất trong study):
   \[
   f(\mathbf{x})=\sum_{i=1}^{p} x_i^2,\quad \mathbf{x}^\star=\mathbf{0}.
   \]

2. **PA (Point Aiming)** (unimodal, tổng quát hoá Sphere khi optimum không ở gốc):
   \[
   f(\mathbf{x})=\|\mathbf{x}-\mathbf{x}^\star\|_2.
   \]
   Paper dùng **2** biến thể \(\mathbf{x}^\star=\mathbf{1}\) và \(\mathbf{x}^\star=10\mathbf{1}\) để đo mức “xa origin”.

3. **CPA (Circular Point Aiming)** (multi-modal, biến thể của PA):
   - Có \(n=5\) điểm mục tiêu \(\mathbf{x}^\star_i\) cùng là **global optimum**.
   - 5 điểm được lấy **cố định cho mỗi \(p\)** bằng cách sample uniform trên đường tròn tâm \(\mathbf{1}\) bán kính 2.
   \[
   f(\mathbf{x})=\min_{i\in\{1,\dots,5\}} \|\mathbf{x}-\mathbf{x}^\star_i\|_2.
   \]

4. **Ackley** (multimodal + non-separable), optimum \(\mathbf{x}^\star=\mathbf{0}\) (paper ghi theo Ackley chuẩn).

5. **Rastrigin** (multimodal, “rugged”), optimum \(\mathbf{x}^\star=\mathbf{0}\).

### 2.3. Quy mô scenario (số bài toán)

Với mỗi hàm, paper thử **4** kích thước không gian:
\[
p\in\{20,100,200,500\}.
\]

Tổng số bài toán S1:
\[
6 \times 4 = 24,
\]
với 6 “tên bài” là: Sphere, PA-1, PA-10, CPA, Ackley, Rastrigin (mỗi tên × 4 giá trị \(p\)). Ký hiệu: `name-p`.

---

## 3) Scenario 2 (S2): ANN regression trên dataset thực (Section 2.2.2)

### 3.1. Mục tiêu bài toán

Cho dataset \(D=\{(\mathbf{x}^{(i)},\tilde{y}^{(i)})\}\) với \(\mathbf{x}^{(i)}\in\mathbb{R}^m\), \(\tilde{y}^{(i)}\in\mathbb{R}\). Tìm hàm \(g:\mathbb{R}^m\to\mathbb{R}\) để tối thiểu hoá MSE:
\[
f(g)=\mathrm{MSE}(g,D)=\frac{1}{n}\sum_{i=1}^{n}\left(g(\mathbf{x}^{(i)})-\tilde{y}^{(i)}\right)^2.
\]

Paper cố định kiến trúc ANN và chỉ tối ưu trọng số \(\boldsymbol{\theta}\).

### 3.2. Kiến trúc ANN + kích hoạt

- **Feed-forward fully connected**, **1 hidden layer**, **1 output**.
- Số neuron input = \(m\), output = 1.
- Hidden layer có kích thước tương đối theo input: \(\rho_h m\) với \(\rho_h\in\{1,2,3\}\) (tức hidden = \(m\), \(2m\), \(3m\)).
- Mọi neuron dùng **tanh**.
- Số tham số:
\[
p=|\boldsymbol{\theta}|=(m+1)\rho_h m + (\rho_h m + 1).
\]

### 3.3. Dataset + tiền xử lý (theo paper)

Paper chọn 3 dataset theo tinh thần benchmark GP của White et al. (2013):

- **Concrete**: \(m=8\), \(|D|=825\).
- **Energy**: \(m=8\), \(|D|=615\).
- **Wine**: \(m=11\), \(|D|=3919\).

Tiền xử lý:

- Chuẩn hoá các biến độc lập \(\mathbf{x}^{(i)}\).
- Rescale nhãn \(\tilde{y}^{(i)}\) vào \([-1,1]\) để phù hợp tanh.

### 3.4. Quy mô scenario

\[
3\ (\text{datasets}) \times 3\ (\rho_h)=9\ \text{problems}.
\]

Paper báo cáo \(p\) dao động từ **81** (Concrete/Energy với \(\rho_h=1\)) tới **430** (Wine với \(\rho_h=3\)). Ký hiệu: `DatasetName-ρh` (ví dụ Concrete-3).

---

## 4) Scenario 3 (S3): điều khiển robot 2D navigation bằng ANN (Section 2.2.3)

### 4.1. Mục tiêu điều khiển + cảm biến

- Robot differential-drive, hình tròn, 2 bánh; 2 đầu ra điều khiển vận tốc 2 bánh.
- Cảm biến:
  - \(m\) proximity sensors phân bố đều trên **nửa cung phía trước**,
  - thêm 2 cảm biến khoảng cách + góc tới target.

Policy \(g:\mathbb{R}^{m+2}\to\mathbb{R}^2\) được thể hiện bằng ANN; tối ưu \(\boldsymbol{\theta}\in\mathbb{R}^p\).

### 4.2. Môi trường mô phỏng (thông số vật lý paper ghi rõ)

- Arena: \(1\text{m}\times 1\text{m}\).
- Bán kính robot: \(5\text{cm}\).
- Proximity sensor range: \(1\text{m}\) (1 khi không có vật cản trong range; 0 khi tiếp xúc trực tiếp).
- Tốc độ tối đa: \(10\text{cm/s}\).
- Khởi tạo ngẫu nhiên (uniform):
  - \(x_0\sim U([0.45,0.55])\),
  - \(y_0\sim U([0.80,0.85])\).
- Target cố định: \((0.50,0.15)\).
- Mô phỏng rời rạc: \(\Delta t=0.1\text{s}\), thời lượng **60s** → fitness **stochastic** theo seed/khởi tạo.

### 4.3. Ba arena (Figure 1)

- Small / Large / Maze (mê cung 2 rào cản).

### 4.4. Kiến trúc ANN + quy mô

Với mỗi \(m\in\{3,5,9\}\):

- Input layer: \(m+2\)
- Hidden layer: \(3(m+2)\) neurons
- Output: 2 neurons
- Activation: **tanh**

Fitness \(f\) (minimize): **khoảng cách trung bình** robot–target trong suốt simulation.

Số bài toán:
\[
3\ (\text{arenas}) \times 3\ (m)=9.
\]

Paper báo cáo \(p\in\{122,212,464\}\) tương ứng các setup. Ký hiệu: `ArenaName-m`.

---

## 5) Scenario 4 (S4): điều khiển VSR (2D voxel-based soft robot) bằng ANN (Section 2.2.4)

### 5.1. Đối tượng: morphology + controller

- VSR: các voxel (ô vuông 2D) nối bằng lò xo–giảm chấn (mô hình theo paper [Medvet et al. 2020] trong reference list của bài).
- Paper chỉ tối ưu **controller (brain)**, morphology cố định:
  - **Biped**: 10 voxel
  - **Tower**: 14 voxel

### 5.2. Cảm biến + chuẩn hoá

Các loại sensor paper liệt kê: proximity (theo hướng \(\alpha\), range \(r\)), area ratio, velocity (theo trục), và sensor “sin” (1 Hz). Có **2** cấu hình sensor:

- **Homogeneous**: area + velocity (cả 2 trục) trên mỗi voxel.
- **Heterogeneous**: phân bố sensor khác nhau theo vùng (head/top/bottom) với các tham số \((r,\alpha)\) paper nêu cụ thể.

Mọi sensor output được normalize về \([-1,1]\).

### 5.3. Ba kiểu controller (C / HoD / HeD)

- **C (centralized)**: một ANN đọc toàn bộ sensor readings, xuất contraction cho mỗi voxel.
- **HoD (homo-distributed)**: mỗi voxel một ANN nhưng **cùng** kiến trúc và **cùng** tham số (weight-sharing).
- **HeD (hetero-distributed)**: mỗi voxel một ANN **khác** tham số; vector tìm kiếm là **concat** tất cả tham số.

**Ràng buộc tương thích (paper nêu rõ)**:

- HoD **không tương thích** heterogeneous sensor config (vì số sensor mỗi voxel không đồng nhất).
- Do đó:
  - Với **heterogeneous sensors**: dùng **C** và **HeD**.
  - Với **homogeneous sensors**: dùng **HoD**.

### 5.4. Kiến trúc ANN + “invoke policy” + noise

- Mỗi ANN: **1 hidden layer**, hidden size **bằng** input size, **tanh**.
- Thêm Gaussian noise variance **0.05** lên mỗi sensor reading.
- Controller chỉ được gọi mỗi **0.2s** (giữ output constant giữa các lần gọi) để tránh rung.

### 5.5. Năm task + fitness (min/max)

1. **Locomotion** (3 terrain: flat / hilly / steppy): maximize \(\bar{v}_x\) của COM trong **30s**.
2. **Jump**: maximize **độ cao tối đa** COM so với mặt đất trong **10s**, **bỏ 5s đầu**.
3. **Balance** (Tower trên swing 30m): minimize **góc trung bình** của swing trong **30s**, có **malus** mỗi timestep swing chạm đất.

Mô phỏng: time step **1/60 s**, các tham số simulator mặc định.

### 5.6. Quy mô search space \(p\) (paper liệt kê)

Với Biped/Tower và (HoD/C/HeD), paper báo cáo các \(p\) sau:

- Biped: **52 / 280 / 374** tương ứng HoD / C / HeD
- Tower: **52 / 275 / 420** tương ứng HoD / C / HeD

Tổng số problems:
\[
5\ (\text{tasks/controller combos}) \times 3 = 15.
\]

Ký hiệu: `TaskShortName-ControllerType` (paper dùng naming scheme này trong mục tóm tắt).

---

## 6) Chỉ số so sánh + thống kê paper dùng (Section 2.3 + Experiments)

### 6.1. “Final best fitness” của một run

Ở iteration cuối (iteration đầu tiên vượt budget), lấy individual tốt nhất trong population theo đúng mục tiêu problem (**min hoặc max**).

### 6.2. NER (Normalized Effectiveness Rank)

Cho **một problem** và \(n_{\mathrm{EAs}}\) EA:

1. Mỗi EA chạy \(n_{\mathrm{rep}}\) lần → thu \(n_{\mathrm{EAs}}n_{\mathrm{rep}}\) giá trị final best fitness.
2. Sort toàn bộ theo mục tiêu problem.
3. Gán rank (1 tốt nhất … \(n_{\mathrm{EAs}}n_{\mathrm{rep}}\) tệ nhất).
4. Chuẩn hoá rank → **NER \(\in[0,1]\)**, **0 là tốt nhất** (theo paper).

### 6.3. NoVS (Number of Victories Score)

Cho **một problem**:

1. Xác định “best EA” = EA có **median** final best fitness tốt nhất.
2. Một EA được tính “chiến thắng” nếu phân bố final best fitness của nó **không khác biệt có ý nghĩa thống kê** so với best EA.
3. Kiểm định: **Wilcoxon signed-rank**, mức ý nghĩa **\(\alpha=0.01\)** (paper cũng nêu lý do kiểm soát family-wise error khi có nhiều pairwise tests).
4. **NoVS của EA trong một scenario** = số problems mà EA đạt “victory”.

**Lưu ý thực hành thống kê**: Wilcoxon paired yêu cầu các EA chạy trên **cùng tập seed** (paper nhấn mạnh việc đổi seed giữa các runs).

### 6.4. EtTQ (Evals to Third Quartile)

Cho **một problem**:

1. Tính quartile thứ 3 \(f_{75}\) trên toàn bộ \(n_{\mathrm{EAs}}n_{\mathrm{rep}}\) final best fitness (sort theo goal min/max).
2. Với mỗi run, đếm số evaluations tối thiểu để đạt fitness **không tệ hơn** \(f_{75}\).
3. Nếu run không bao giờ đạt ngưỡng đó → **EtTQ = \(n_{\mathrm{eval}}=10\,000\)**.

EtTQ **càng nhỏ càng tốt**.

---

## 7) Quy mô thí nghiệm (số runs) + môi trường tính toán (Section 3 mở đầu)

### 7.1. Số problems tổng

\[
24\ (\text{S1}) + 9\ (\text{S2}) + 9\ (\text{S3}) + 15\ (\text{S4}) = 57.
\]

### 7.2. Số runs / (EA, problem)

- S1, S2, S3: \(n_{\mathrm{rep}}=30\).
- S4: \(n_{\mathrm{rep}}=20\) (nặng hơn).

Tổng runs:
\[
9\times(24+9+9)\times 30 \;+\; 9\times 15\times 20
= 9\times 42\times 30 + 9\times 15\times 20
= 14\,040.
\]

### 7.3. Phần mềm + phần cứng (paper báo cáo)

- Framework: **JGEA**.
- Máy: Ubuntu 20.04; Intel Xeon W-2295 @ 3.0 GHz, 36 cores, 64 GB RAM (**2 máy**, dùng **không độc quyền** → paper nói không tập trung thảo luận wall-clock time).

Paper cũng báo cáo thứ tự độ lớn thời gian/run (minh hoạ):

- S1: ~1–5s
- S2: ~75s (Concrete/Energy), ~250s (Wine)
- S3: ~100–200s
- S4: ~10,000s (balance/locomotion), ~2,000s (jump)

---

## 8) Kết quả tổng hợp mà tác giả báo cáo (điểm nhấn + bảng NoVS)

Phần này là **thống kê/kết luận trong paper** (không phải “chạy lại” trong repo).

### 8.1. Bảng NoVS theo scenario (Table 1 trong paper; paper_summary lưu lại như sau)

| EA | Synthetic (24) | Regression (9) | 2D nav. (9) | VSR (15) |
|---|---:|---:|---:|---:|
| CMA-ES | **18** | 0 | 0 | 5 |
| DE | 0 | **7** | **9** | **13** |
| PSO | 0 | 1 | 2 | 2 |
| ES-0.02 | 2 | 3 | 0 | 0 |
| ES-0.25 | 0 | 0 | 3 | 1 |
| ES-0.5 | 0 | 0 | 3 | 3 |
| GA-0.02 | 6 | 4 | 0 | 1 |
| GA-0.25 | 0 | 0 | 4 | 11 |
| GA-0.5 | 0 | 0 | 5 | 12 |

### 8.2. Điểm nhấn theo từng scenario (theo mục kết quả của paper)

- **S1 (synthetic)**: CMA-ES mạnh nhất về effectiveness + efficiency; NoVS CMA-ES **18/24**. Trên Ackley/Rastrigin (rugged) khi \(p\) lớn, CMA-ES có thể suy giảm; **GA-0.02** có NoVS **6** (4×Rastrigin + Ackley-500 + Sphere-500). Ảnh hưởng của \(p\) lên ranking thường **nhỏ** hơn ruggedness (paper thảo luận kèm Figure 3–4).
- **S2 (regression)**: CMA-ES “tệ nhất nhóm” với **NoVS=0**; **DE** dẫn với **NoVS=7/9**. Về efficiency, **GA-0.02** và **ES-0.02** có EtTQ tốt; nhóm {GA nhỏ σ, ES nhỏ σ, DE, PSO} có phân phối EtTQ hẹp hơn nhóm còn lại (paper mô tả kèm Figure 5–6).
- **S3 (2D navigation)**: chênh lệch giữa EA **ít rõ** hơn S1/S2; **DE** vẫn best với **NoVS=9/9**; **ES-0.02** thường kém và không ổn định (Figure 7–8).
- **S4 (VSR)**: có cả maximize và minimize tasks; **DE** và **GA-0.5** mạnh về effectiveness; về efficiency **DE** và **CMA-ES** nổi bật hơn GA-0.5. **DE** đạt **13/15** NoVS. Paper cũng phân tích “discontinuity” NER khi đổi task/morphology dẫn tới thay đổi \(p\) (Tower vs Biped) và thảo luận CMA-ES khó với **HoD** (Figure 9–10).

### 8.3. Thông điệp discussion chính (paper Section 3.7)

- Trên các bài neuroevolution (S2–S4), **DE** luôn có **NoVS lớn nhất**, theo sau là các biến thể GA/ES “đơn giản”.
- Hiệu năng trên **synthetic** **không dự đoán** hiệu năng trên neuroevolution phức tạp hơn.
- Paper đưa giả thuyết: độ phức tạp/độ gián tiếp của mapping \(\boldsymbol{\theta}\mapsto f(\boldsymbol{\theta})\) liên quan tới ruggedness; trên landscape rất gồ ghề, GA với **σ lớn** + **crossover** (ở DE/GA) có lợi thế khám phá.

---

## 9) Hình trong paper: loại biểu đồ và ý nghĩa (thống kê / trực quan)

Phần này tóm tắt **Figure 1–10** và **Table 1** theo đúng vai trò trong bài: một nhóm hình **minh hoạ bài toán**, một nhóm **lặp lại cùng “cú pháp”** cho từng scenario để đọc effectiveness + efficiency + ảnh hưởng của \(p\).

### 9.1. Hình bối cảnh (không phải so sánh EA)

| Hình | Nội dung | Ý nghĩa khi đọc paper |
|------|----------|------------------------|
| **Figure 1** | Ba arena điều hướng 2D (Small / Large / Maze): vùng xuất phát, target, chướng ngại. | Giúp hiểu **độ khó** và bối cảnh **S3**; không phải thống kê thuật toán. |
| **Figure 2** | Các task VSR (locomotion trên nhiều địa hình, jump, balance) và morphology Biped / Tower. | Giúp hiểu **S4**, fitness maximize vs minimize, robot nào dùng cho task nào. |

### 9.2. Bốn cặp hình kết quả — cùng một cấu trúc cho S1 → S4

Paper dùng **visual syntax** lặp lại:

#### (A) Phân phối **final best fitness** — Figure **3, 5, 7, 9**

- **Loại**: phân phối \(f(\mathbf{x}^\*)\) hoặc \(f(\boldsymbol{\theta}^\*)\) trên các run (**30** run cho S1–S3, **20** run cho S4), **mỗi problem một panel**, trong panel là phân phối theo từng EA (paper mô tả là *distributions* — thực tế là dạng **box plot / violin** tương đương).
- **Ý nghĩa**:
  - So sánh **chất lượng tuyệt đối** *trong cùng một problem*: median, độ rộng, đuôi → ổn định hay biến động giữa các seed.
  - Fitness **không so trực tiếp giữa các problem khác nhau**; vì vậy các hình lẻ chủ yếu để đọc **chi tiết từng bài**.

#### (B) **NER**, **EtTQ**, và **NER theo \(p\)** — Figure **4, 6, 8, 10**

Theo caption paper (ví dụ Figure 4 / 6 / 8 / 10), mỗi figure này gồm:

1. **Phân phối NER** trên toàn bộ run trong scenario — NER đã **chuẩn hoá xếp hạng** giữa các EA trên từng problem rồi gom lại, \(\mathrm{NER}\in[0,1]\), **càng thấp càng tốt** → trả lời: *EA nào **tương đối** tốt khi so công bằng qua nhiều problem?*
2. **Phân phối EtTQ** — số evaluation để đạt ngưỡng “đủ tốt” theo định nghĩa quartile thứ 3 (Methods); **càng thấp càng tốt**; không đạt thì EtTQ = \(n_{\mathrm{eval}}\) → trả lời: *EA nào **hiệu quả về chi phí tính toán** (trong budget 10 000)?*
3. **Đồ thị NER theo \(p\)** — NER **trung bình** theo các problem cùng chiều \(p\) của vector tham số → trả lời: *khi **tăng số chiều** \(p\), thứ hạng EA có đổi không?*

**Ý nghĩa chung của cặp hình chẵn (4, 6, 8, 10)**:

- Bổ sung cho hình lẻ: chuyển từ fitness thô sang **NER** (so được giữa các problem) và **EtTQ** (góc efficiency).
- Đoạn **NER vs \(p\)** dùng để thảo luận **dimensionality** so với **ruggedness** (S1: \(p\) ít quyết định ranking hơn landscape; S2: một số EA NER xấu dần khi \(p\) tăng; S3: chênh lệch mềm hơn; S4: NER không “flat” theo \(p\), có **gián đoạn** giữa các problem khác task/morphology — paper giải thích rõ ví dụ Tower+Balance vs Biped+locomotion).

**Ánh xạ nhanh scenario ↔ figure**:

| Scenario | Fitness distributions | NER + EtTQ + NER vs \(p\) |
|----------|----------------------|---------------------------|
| S1 Synthetic | **Figure 3** | **Figure 4** |
| S2 Regression | **Figure 5** | **Figure 6** |
| S3 2D navigation | **Figure 7** | **Figure 8** |
| S4 VSR | **Figure 9** | **Figure 10** |

### 9.3. Table 1 — NoVS (tổng hợp thống kê, không phải biểu đồ)

- **Table 1** ghi **NoVS** của mỗi EA trên **từng scenario** (số problem mà EA “thắng” theo nghĩa **Wilcoxon** so với EA median-best, \(\alpha=0.01\)).
- **Ý nghĩa**: một con số **tổng hợp có kiểm định** bên cạnh việc “nhìn hình”; phù hợp khi cần so sánh nhanh **ai thắng nhiều problem** trong scenario, trong khi Figure 3–10 cho **phân phối** và **xu hướng theo \(p\)**.

---

## 10) Ghi chú về “CSV/log” (để khỏi nhầm với dataset)

Trong các artifact đi kèm paper/repo gốc, các file CSV thường là **log kết quả chạy** (fitness theo evaluations/iterations), **không** thay thế dataset UCI cho S2. Dataset S2 paper nêu rõ là Concrete/Energy/Wine như mục 3.3.
