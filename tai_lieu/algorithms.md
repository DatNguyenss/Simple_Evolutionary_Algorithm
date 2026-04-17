# Algorithms — Pseudocode, công thức, tham số, ví dụ (CMA-ES / DE / PSO / ES / GA)

Tài liệu này giúp bạn **nắm vững 5 thuật toán tiến hoá/tối ưu** hay gặp trong neuroevolution và tối ưu số học:

- **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy)
- **DE/rand/1/bin** (Differential Evolution — biến thể kinh điển)
- **PSO** (Particle Swarm Optimization)
- **ES (basic)** (Evolution Strategy cơ bản — sampling quanh trung bình)
- **GA số thực** (Genetic Algorithm cho \(\mathbb{R}^p\))

Mỗi thuật toán có:
- **Pseudocode** (đọc như thuật toán “có thể code được”)
- **Công thức toán** (chuẩn kiểu MathType/LaTeX)
- **Bảng tham số** (ý nghĩa + gợi ý chọn)
- **Ví dụ minh hoạ số nhỏ** (1–2 chiều) để bạn thấy *một bước cập nhật xảy ra thế nào*

---

## 0) Ký hiệu chung (dùng xuyên suốt)

- **Bài toán tối ưu**: thường là *minimize*

\[
\min_{\mathbf{x}\in \mathbb{R}^p} f(\mathbf{x})
\]

- \(p\): số chiều của nghiệm.
- \(\mathbf{x}\): một nghiệm (vector).
- \(n_{\text{pop}}\): kích thước quần thể.
- \(\mathcal{N}(\boldsymbol{\mu}, \mathbf{\Sigma})\): phân phối Gaussian đa biến.
- \(\mathcal{U}(a,b)\): phân phối đều.
- Trong ví dụ minh hoạ, ta dùng hàm Sphere 1D/2D:

\[
f(\mathbf{x})=\lVert \mathbf{x}\rVert^2
\quad\text{(minimize, optimum tại } \mathbf{0}\text{)}
\]

---

## 1) CMA-ES

### Ý tưởng “nhìn phát hiểu ngay”

Thay vì “lai/đột biến” theo kiểu rời rạc, CMA-ES **học một phân phối Gaussian** để lấy mẫu nghiệm:

- Lúc đầu: Gaussian rộng, khám phá nhiều.
- Càng về sau: Gaussian co lại và “xoay” theo hướng có lợi nhờ **ma trận hiệp phương sai**.

### Công thức cốt lõi (mức đủ dùng để hiểu và triển khai ý tưởng)

Tại iteration \(t\), ta lấy mẫu \(\lambda\) cá thể:

\[
\mathbf{x}^{(k)}_t = \mathbf{m}_t + \sigma_t \mathbf{A}_t \mathbf{z}^{(k)},\quad 
\mathbf{z}^{(k)}\sim \mathcal{N}(\mathbf{0}, \mathbf{I}),\ k=1,\dots,\lambda
\]

Trong đó:
- \(\mathbf{m}_t\): mean (trung tâm phân phối).
- \(\sigma_t\): step-size toàn cục.
- \(\mathbf{A}_t\mathbf{A}_t^\top = \mathbf{C}_t\): phân rã của covariance \(\mathbf{C}_t\).

Sau khi đánh giá \(f(\mathbf{x}^{(k)}_t)\) và sắp theo fitness, CMA-ES cập nhật mean theo **trung bình có trọng số** của top \(\mu\) cá thể:

\[
\mathbf{m}_{t+1} = \sum_{i=1}^{\mu} w_i\, \mathbf{x}^{(i:\lambda)}_t
\quad\text{với}\quad \sum_i w_i = 1,\ w_i>0
\]

*(Các cập nhật “path”, \(\mathbf{C}_t\), \(\sigma_t\) khá dài; khi code bạn thường dùng thư viện/triển khai chuẩn. Với mục tiêu học hiểu, mấu chốt là: **lấy mẫu từ Gaussian và cập nhật Gaussian dựa trên top cá thể**.)*

### Pseudocode

```text
Input: f, p
Init: m ← random in R^p
      σ ← σ0
      C ← I_p
Repeat until budget:
  Sample λ points:
    for k = 1..λ:
      z_k ~ N(0, I)
      x_k ← m + σ * chol(C) * z_k
  Evaluate f(x_k)
  Sort x_k by fitness (ascending if minimize)
  Update mean:
    m ← Σ_{i=1..μ} w_i * x_(i)
  Update C (learn covariance) and σ (adapt step-size)
Return best seen
```

### Bảng tham số (mức thực hành)

| Tham số | Ký hiệu | Vai trò | Gợi ý nhanh |
|---|---:|---|---|
| Population size | \(\lambda\) | số mẫu mỗi iteration | hay dùng default theo \(p\) |
| #parents | \(\mu\) | số mẫu tốt nhất để cập nhật mean | \(\mu \approx \lambda/2\) |
| Weights | \(w_i\) | trọng số cho top mẫu | giảm dần theo rank |
| Step-size | \(\sigma\) | độ “nhảy” toàn cục | để CMA-ES tự thích nghi |
| Covariance | \(\mathbf{C}\) | học hướng/độ tương quan giữa chiều | lõi sức mạnh CMA-ES |

### Ví dụ minh hoạ 1D (trực giác)

Minimize \(f(x)=x^2\). Giả sử tại \(t\):
- \(m_t=2\), \(\sigma_t=1\), \(\lambda=4\).
- Sample \(z\): \([-1.0,\ 0.2,\ 1.5,\ -0.3]\)
- Tạo mẫu: \(x = m + \sigma z \Rightarrow [1.0,\ 2.2,\ 3.5,\ 1.7]\)
- Fitness: \([1.00,\ 4.84,\ 12.25,\ 2.89]\)

Top 2 (giả sử \(\mu=2\), \(w_1=0.6, w_2=0.4\)): \(x_{best} = 1.0\), \(x_{2nd}=1.7\)

\[
m_{t+1} = 0.6\cdot 1.0 + 0.4\cdot 1.7 = 1.28
\]

Bạn thấy mean **dịch về gần 0**. Các cập nhật \(\sigma\) và \(\mathbf{C}\) sẽ làm bước đi ngày càng “chuẩn”.

---

## 2) Differential Evolution (DE/rand/1/bin)

### Ý tưởng

DE tạo nghiệm mới bằng cách:
- Lấy **3 cá thể khác nhau** để tạo “hướng sai phân”.
- Dùng **crossover theo từng chiều** để trộn “vector đột biến” với “vector gốc”.
- Dùng **selection 1–1**: nghiệm mới tốt hơn thì thay nghiệm cũ.

### Công thức

Với mỗi target \(\mathbf{x}_i\), chọn ngẫu nhiên \(r_1, r_2, r_3\) khác nhau và khác \(i\).

**Mutation** (rand/1):

\[
\mathbf{v}_i = \mathbf{x}_{r_1} + F(\mathbf{x}_{r_2}-\mathbf{x}_{r_3})
\]

**Binomial crossover**:

\[
u_{i,j}=
\begin{cases}
v_{i,j} & \text{nếu } \mathrm{rand}(0,1) < CR \ \text{hoặc } j=j_{\text{rand}}\\
x_{i,j} & \text{ngược lại}
\end{cases}
\]

**Selection** (minimize):

\[
\mathbf{x}_i \leftarrow 
\begin{cases}
\mathbf{u}_i & \text{nếu } f(\mathbf{u}_i)\le f(\mathbf{x}_i)\\
\mathbf{x}_i & \text{ngược lại}
\end{cases}
\]

### Pseudocode

```text
Input: f, population {x_i}_{i=1..NP}
Repeat:
  for i = 1..NP:
    pick distinct r1, r2, r3 (≠ i)
    v ← x_r1 + F*(x_r2 - x_r3)
    u ← bin_crossover(x_i, v, CR)   # per-dimension
    if f(u) ≤ f(x_i): x_i ← u
Return best seen
```

### Bảng tham số

| Tham số | Ký hiệu | Vai trò | Gợi ý nhanh |
|---|---:|---|---|
| Population size | \(NP\) | số cá thể | nhỏ–vừa tuỳ budget |
| Differential weight | \(F\) | độ mạnh của sai phân | hay \(0.4\)–\(0.9\) |
| Crossover rate | \(CR\) | xác suất lấy gene từ \(\mathbf{v}\) | hay \(0.5\)–\(0.9\) |

### Ví dụ 2D (một bước cho 1 cá thể)

Minimize \(f(\mathbf{x})=\lVert \mathbf{x}\rVert^2\).
Giả sử:
- \(\mathbf{x}_{r_1}=(1,1)\), \(\mathbf{x}_{r_2}=(2,-1)\), \(\mathbf{x}_{r_3}=(0,1)\)
- \(F=0.5\)

\[
\mathbf{v}=(1,1)+0.5\big((2,-1)-(0,1)\big)=(1,1)+0.5(2,-2)=(2,0)
\]

Nếu target \(\mathbf{x}_i=(2,2)\) và crossover tạo \(\mathbf{u}=(2,0)\) (lấy cả 2 chiều từ \(\mathbf{v}\)):
- \(f(\mathbf{x}_i)=8\), \(f(\mathbf{u})=4\) ⇒ thay \(\mathbf{x}_i\leftarrow \mathbf{u}\).

---

## 3) Particle Swarm Optimization (PSO)

### Ý tưởng

Mỗi “hạt” là một nghiệm. Hạt có **vận tốc**:
- bị kéo về **best của chính nó** (cognitive)
- và **best của bầy** (social)

### Công thức cập nhật chuẩn

Với hạt \(i\):

\[
\begin{aligned}
\mathbf{v}_i &\leftarrow \omega \mathbf{v}_i \\
&\quad + c_1 r_1 (\mathbf{p}_i - \mathbf{x}_i) \\
&\quad + c_2 r_2 (\mathbf{g} - \mathbf{x}_i)
\end{aligned}
\]

\[
\mathbf{x}_i \leftarrow \mathbf{x}_i + \mathbf{v}_i
\]

Trong đó:
- \(\mathbf{p}_i\): personal best của hạt \(i\)
- \(\mathbf{g}\): global best của swarm
- \(r_1, r_2 \sim \mathcal{U}(0,1)\) (thường theo từng chiều)

### Pseudocode

```text
Init particles {x_i}, velocities {v_i}
For each particle: p_i ← x_i
g ← best among {p_i}
Repeat:
  for each particle i:
    v_i ← ω*v_i + c1*r1*(p_i - x_i) + c2*r2*(g - x_i)
    x_i ← x_i + v_i
    if f(x_i) better than f(p_i): p_i ← x_i
  g ← best among {p_i}
Return g
```

### Bảng tham số

| Tham số | Ký hiệu | Vai trò | Gợi ý nhanh |
|---|---:|---|---|
| Inertia | \(\omega\) | giữ quán tính vận tốc | lớn: khám phá, nhỏ: hội tụ |
| Cognitive | \(c_1\) | kéo về best cá nhân | thường ~1–2 |
| Social | \(c_2\) | kéo về best toàn cục | thường ~1–2 |
| Population | \(n_{\text{pop}}\) | số hạt | tuỳ budget |

### Ví dụ 1D (một bước)

Minimize \(f(x)=x^2\).
Giả sử:
- \(x=3\), \(v=-1\)
- \(p=2\) (best cá nhân), \(g=1\) (best toàn cục)
- \(\omega=0.5, c_1=c_2=1.5\)
- \(r_1=0.4, r_2=0.2\)

\[
v \leftarrow 0.5(-1) + 1.5(0.4)(2-3) + 1.5(0.2)(1-3)
= -0.5 -0.6 -0.6 = -1.7
\]
\[
x \leftarrow 3 + (-1.7) = 1.3
\]
Fitness giảm mạnh: \(9 \to 1.69\).

---

## 4) ES (basic) — biến thể cơ bản “mean + noise”

### Ý tưởng

ES basic trong paper (bản đơn giản) làm đúng 3 bước:
1) lấy top parents  
2) tính trung bình \(\boldsymbol{\mu}\)  
3) sample quần thể mới quanh \(\boldsymbol{\mu}\) bằng Gaussian nhiễu độc lập theo từng chiều.

### Công thức

Chọn \(n_{\text{parents}}\) nghiệm tốt nhất: \(\{\mathbf{x}^{(1)},\dots,\mathbf{x}^{(n_{\text{parents}})}\}\).

\[
\boldsymbol{\mu} = \frac{1}{n_{\text{parents}}}\sum_{i=1}^{n_{\text{parents}}}\mathbf{x}^{(i)}
\]

Tạo cá thể mới:

\[
\mathbf{x}' \sim \mathcal{N}(\boldsymbol{\mu}, \sigma \mathbf{I})
\quad\Leftrightarrow\quad
\mathbf{x}' = \boldsymbol{\mu} + \boldsymbol{\varepsilon},\ 
\boldsymbol{\varepsilon}\sim \mathcal{N}(\mathbf{0}, \sigma\mathbf{I})
\]

### Pseudocode

```text
Init population P (size n_pop)
Repeat:
  Evaluate f for all x in P
  Parents ← best n_parents in P
  μ ← mean(Parents)
  NewP ← {best(P)}    # elitism 1 cá thể
  while |NewP| < n_pop:
    sample x ~ N(μ, σ I)
    add x to NewP
  P ← NewP
Return best seen
```

### Bảng tham số

| Tham số | Ký hiệu | Vai trò | Gợi ý nhanh |
|---|---:|---|---|
| Population size | \(n_{\text{pop}}\) | số cá thể | 20–100 tuỳ bài |
| #parents | \(n_{\text{parents}}\) | số chọn lọc để tính mean | ~\(0.2\)–\(0.5\) \(n_{\text{pop}}\) |
| Step-size | \(\sigma\) | độ nhiễu khi sample quanh mean | quan trọng nhất |
| Elitism | — | giữ best | giúp ổn định |

### Ví dụ 1D

Minimize \(f(x)=x^2\). Population 5 cá thể: \([2,\ -1,\ 0.5,\ 3,\ -2]\).
Chọn \(n_{\text{parents}}=2\) best theo fitness: \([-1,\ 0.5]\).

\[
\mu = (-1 + 0.5)/2 = -0.25
\]

Lấy mẫu quanh \(\mu\) (ví dụ \(\sigma=0.25\)): tạo cá thể mới gần \(-0.25\), nên fitness kỳ vọng tốt hơn ban đầu.

---

## 5) GA số thực (mutation Gaussian + segment geometric crossover)

### Ý tưởng

GA tạo thế hệ mới bằng:
- **Selection** (tournament): chọn bố/mẹ từ quần thể dựa trên fitness.
- **Variation**: hoặc **mutation** (thêm nhiễu Gaussian) hoặc **crossover** (trộn 2 bố mẹ theo một công thức tuyến tính).
- **Replacement**: trộn cha mẹ + con, giữ top \(n_{\text{pop}}\).

### Công thức

**Gaussian mutation**:

\[
\mathbf{x}' = \mathbf{x} + \boldsymbol{\varepsilon},\quad
\boldsymbol{\varepsilon}\sim \mathcal{N}(\mathbf{0}, \sigma\mathbf{I})
\]

**Segment geometric crossover** (như paper mô tả):

\[
\mathbf{x}' = \mathbf{x}_1 + \alpha(\mathbf{x}_2 - \mathbf{x}_1) + \boldsymbol{\varepsilon},
\quad \alpha\sim \mathcal{U}([0,1]),\ 
\boldsymbol{\varepsilon}\sim \mathcal{N}(\mathbf{0}, \sigma\mathbf{I})
\]

### Pseudocode

```text
Input: population P (size n_pop)
Repeat:
  Offspring ← ∅
  for k = 1..n_pop:
    if rand < p_xo:
      x1 ← tournament_select(P, n_tour)
      x2 ← tournament_select(P, n_tour)
      α ~ U(0,1)
      ε ~ N(0, σ I)
      child ← x1 + α*(x2 - x1) + ε
    else:
      x ← tournament_select(P, n_tour)
      ε ~ N(0, σ I)
      child ← x + ε
    add child to Offspring
  P ← best n_pop from (P ∪ Offspring)   # elitism qua merge+truncate
Return best seen
```

### Bảng tham số

| Tham số | Ký hiệu | Vai trò | Gợi ý nhanh |
|---|---:|---|---|
| Population size | \(n_{\text{pop}}\) | số cá thể | 50–200 tuỳ budget |
| Crossover prob. | \(p_{\text{xo}}\) | tỉ lệ sinh con bằng crossover | hay 0.6–0.9 |
| Tournament size | \(n_{\text{tour}}\) | mức “tham lam” khi chọn lọc | lớn: khai thác, nhỏ: khám phá |
| Step-size | \(\sigma\) | mức nhiễu khi mutation/crossover | rất quan trọng |

### Ví dụ 2D (crossover)

Minimize \(f(\mathbf{x})=\lVert \mathbf{x}\rVert^2\).
Cho \(\mathbf{x}_1=(4,0)\), \(\mathbf{x}_2=(0,2)\), \(\alpha=0.25\), và bỏ nhiễu \(\boldsymbol{\varepsilon}=\mathbf{0}\) để dễ nhìn:

\[
\mathbf{x}' = (4,0)+0.25\big((0,2)-(4,0)\big) = (4,0)+0.25(-4,2) = (3,0.5)
\]
Fitness: \(f(\mathbf{x}_1)=16\), \(f(\mathbf{x}_2)=4\), \(f(\mathbf{x}')=9.25\) ⇒ con nằm “giữa” 2 bố mẹ theo đoạn thẳng nối chúng.

---

## 6) Bảng tóm tắt nhanh: “dùng khi nào?”

- **CMA-ES**: mạnh khi landscape “mịn”, có cấu trúc tương quan giữa chiều; học covariance tốt.
- **DE**: rất mạnh trong tối ưu số thực, đặc biệt khi crossover theo chiều giúp “nhảy xa” hợp lý.
- **PSO**: dễ hiểu, dễ chạy, nhạy tham số; hợp bài tối ưu liên tục vừa–nhỏ.
- **ES basic**: baseline rất rõ ràng; hiệu năng phụ thuộc mạnh vào \(\sigma\).
- **GA số thực**: linh hoạt; crossover giúp khám phá landscape gồ ghề (đặc biệt trong control/neuroevolution).

---

## 7) Bài tập tự kiểm tra (để bạn nắm thật)

1) Với DE, giải thích vì sao \(F\) quá lớn dễ “văng” khỏi vùng tốt.  
2) Với PSO, nếu \(\omega=0\) thì điều gì xảy ra? Nếu \(\omega\) quá lớn thì sao?  
3) Với ES/GA, \(\sigma\) lớn và \(\sigma\) nhỏ tương ứng exploration/exploitation thế nào?  
4) Với CMA-ES, trực giác của “covariance xoay theo hướng tốt” là gì? (gợi ý: ellipsoid sampling)

