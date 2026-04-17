# Ý tưởng cải tiến (đề xuất để đạt “NOTE 2”)

Tài liệu này đề xuất **một cải tiến nhỏ nhưng có thể kiểm chứng định lượng** so với các EA trong paper *El Saliby et al. (2024)*, sử dụng đúng pipeline đánh giá (NER / EtTQ / NoVS).

---

## 1) Bối cảnh và động cơ

Paper so sánh 9 cấu hình EA (CMA-ES, DE, PSO, ES-σ, GA-σ) với:

- **Budget**: dừng tại iteration đầu tiên mà số fitness evaluations vượt \(n_{eval}=10\,000\)
- **Init**: \(U([-1,1])^p\)
- **Metrics**:
  - **Effectiveness**: NER (Normalized Effectiveness Rank) của final best fitness
  - **Efficiency**: EtTQ (Evals to Third Quartile)
  - **Tổng hợp**: NoVS (Wilcoxon signed-rank test, \(\alpha=0.01\))

Trong paper, **ES** và **GA** chỉ khác nhau ở \(\sigma \in \{0.02,0.25,0.5\}\) nhưng **\(\sigma\) là hyperparameter cố định theo run**. Trên các landscape gồ ghề, “step-size” tối ưu thường **không cố định**: cần lớn lúc đầu để khám phá và nhỏ dần để tinh chỉnh.

---

## 2) Cải tiến đề xuất: Adaptive-σ cho ES và GA (không tăng budget)

### 2.1. Mục tiêu

- Giữ nguyên framework đánh giá của paper (NER/EtTQ/NoVS)
- Không tăng budget \(n_{eval}\)
- Chỉ thay đổi **cách chọn \(\sigma\) theo thời gian** trong ES/GA

### 2.2. Ý tưởng cốt lõi

Thay vì \(\sigma\) cố định, dùng **lịch trình giảm \(\sigma\)** theo tỉ lệ budget đã tiêu thụ:

\[
\sigma(t) = \sigma_{\max}\cdot\left(\frac{\sigma_{\min}}{\sigma_{\max}}\right)^{t}
\quad\text{với}\quad
t=\frac{\text{evals}}{n_{eval}}\in[0,1]
\]

Trong đó:

- \(\sigma_{\max}=0.5\) (khám phá mạnh lúc đầu)
- \(\sigma_{\min}=0.02\) (tinh chỉnh cuối run)

Đây là dạng **exponential annealing** (giảm theo cấp số nhân), thường ổn định hơn giảm tuyến tính.

### 2.3. Áp dụng vào ES (basic)

ES paper: mỗi iteration lấy mean \(\mu\) của top parents, sinh \(n_{pop}-1\) điểm từ \( \mathcal{N}(\mu,\sigma I)\) + giữ best.

Cải tiến: thay \(\sigma\) bằng \(\sigma(t)\), với \(t\) tính từ `evals/n_eval`.

Kỳ vọng:

- Giảm nguy cơ “kẹt” vì \(\sigma\) quá nhỏ ngay từ đầu
- Giảm rung/lệch khi \(\sigma\) quá lớn ở cuối run

### 2.4. Áp dụng vào GA (mutation + segment geometric crossover)

GA paper: mỗi offspring là mutation hoặc crossover và đều có nhiễu Gaussian \( \epsilon\sim\mathcal{N}(0,\sigma I)\).

Cải tiến: dùng \(\sigma(t)\) để scale nhiễu theo tiến trình:

- Lúc đầu: nhiễu lớn để nhảy xa (exploration)
- Lúc cuối: nhiễu nhỏ để fine-tune (exploitation)

---

## 3) Định nghĩa “cải tiến thành công” (để đạt NOTE 2)

Một cải tiến được coi là “thành công” nếu khi benchmark trên cùng scenario:

- **NER median** thấp hơn baseline tương ứng (ES-0.02/0.25/0.5 hoặc GA-0.02/0.25/0.5)
- Và/hoặc **EtTQ median** thấp hơn
- Và quan trọng nhất: **NoVS tăng** (thắng nhiều problem hơn theo Wilcoxon paired, \(\alpha=0.01\))

Khuyến nghị báo cáo theo đúng style paper:

- Boxplot NER, EtTQ cho toàn scenario
- NER vs \(p\)
- Table NoVS (baseline vs improved)

---

## 4) Thiết kế thí nghiệm / Ablation tối thiểu

### 4.1. Baselines (từ paper)

- ES-0.02, ES-0.25, ES-0.5
- GA-0.02, GA-0.25, GA-0.5

### 4.2. Phương pháp mới

- ES-adaptive-anneal(\(\sigma_{\max}=0.5,\sigma_{\min}=0.02\))
- GA-adaptive-anneal(\(\sigma_{\max}=0.5,\sigma_{\min}=0.02\))

### 4.3. Ablation (nhẹ nhưng đủ “khoa học”)

Chạy thêm 2 biến thể để chứng minh “cái gì tạo ra lợi ích”:

1. **Adaptive chỉ ở ES**, GA giữ nguyên
2. **Adaptive chỉ ở GA**, ES giữ nguyên

Nếu muốn gọn hơn: chỉ cần 1 trong 2 (ưu tiên GA vì paper nhấn mạnh GA/DE trên landscape gồ ghề).

---

## 5) Rủi ro và cách giảm rủi ro

- **Rủi ro 1**: Adaptive làm tăng variance giữa seeds.
  - Giảm rủi ro bằng cách giữ nguyên seed list, báo cáo phân phối NER/EtTQ.
- **Rủi ro 2**: Annealing không hợp mọi problem.
  - Thử thêm lịch trình tuyến tính như ablation phụ:
    \(\sigma(t)=\sigma_{\max}-(\sigma_{\max}-\sigma_{\min})t\).

---

## 6) Tóm tắt 1 câu

Đề xuất **giữ nguyên paper setup**, chỉ thay \(\sigma\) cố định bằng **adaptive-σ annealing** cho ES/GA để tăng khả năng khám phá sớm và tinh chỉnh muộn, rồi chứng minh bằng **NER/EtTQ/NoVS** theo đúng evaluation đã reproduce.

