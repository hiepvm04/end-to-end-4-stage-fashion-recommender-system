# Hands-on Personalized Recommender System

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![Dependencies](https://img.shields.io/badge/dependencies-uv-success)
![Platform](https://img.shields.io/badge/platform-Hopsworks-orange)
![Status](https://img.shields.io/badge/status-Personal%20Project-purple)

## Bối cảnh dự án (Project Context)

Dự án được xây dựng với mục đích học tập, thực hành và áp dụng các kiến thức về Học máy (Machine Learning), Hệ thống gợi ý (Recommender Systems) và MLOps vào thực tế. Dự án mô phỏng một pipeline học máy hoàn chỉnh từ khâu xử lý dữ liệu đến triển khai mô hình.

## Mô tả dự án (Project Description)

**Hands-on Personalized Recommender System** là hệ thống gợi ý sản phẩm thời trang được cá nhân hóa hoàn chỉnh (end-to-end). Dự án áp dụng **Kiến trúc Gợi ý 4 Giai đoạn (4-Stage Recommender Architecture)** bao gồm:
1. **Retrieval (Truy xuất):** Sử dụng mô hình Two-Tower Embedding để tìm kiếm các mặt hàng tiềm năng.
2. **Filtering (Lọc):** Loại bỏ các sản phẩm không phù hợp với ngữ cảnh hiện tại.
3. **Scoring/Ranking (Xếp hạng):** Sử dụng mô hình Gradient Boosting (CatBoost) hoặc LLM (thông qua OpenAI) để chấm điểm và xếp hạng.
4. **Ordering (Sắp xếp):** Sắp xếp và hiển thị danh sách cuối cùng cho người dùng thông qua giao diện Streamlit.

## Trợ giúp trực quan (Visuals)

**1. Giao diện ứng dụng Streamlit (UI Example):**
![UI Example](assets/ui_example.png)

**2. Kiến trúc Hệ thống Tổng thể (System Architecture):**
![System Architecture](assets/system_architecture.png)

**3. Kiến trúc Gợi ý 4 Giai đoạn:**
![4 Stage Recommender Architecture](assets/4_stage_recommender_architecture.png)

**4. Mô hình Two-Tower:**
![Two Tower Model](assets/two_tower_embedding_model.png)

## Hướng dẫn cài đặt (Installation)

Dự án sử dụng trình quản lý gói `uv` để quản lý môi trường ảo và cài đặt thư viện một cách nhanh chóng.

**Bước 1: Clone kho lưu trữ repository**
```bash
git clone <https://github.com/hiepvm04/end-to-end-4-stage-fashion-recommender-system.git>
cd end-to-end-4-stage-fashion-recommender-system
```

**Bước 2: Cài đặt Python 3.11+ (nếu chưa có sẵn)**
```bash
make install-python
```

**Bước 3: Khởi tạo môi trường ảo và cài đặt thư viện**
```bash
make install
```

**Bước 4: Cấu hình biến môi trường**
Sao chép tệp mẫu và cập nhật các khóa API của bạn để kết nối với các dịch vụ bên ngoài:
```bash
cp .env.example .env
```
Mở tệp `.env` và điền thông tin:
```env
HOPSWORKS_API_KEY="your_hopsworks_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here" # Cần thiết nếu bạn muốn thử nghiệm tính năng LLM Ranking
```

## Cách dùng và ví dụ (Usage & Examples)

Bạn có thể chạy toàn bộ pipeline theo từng bước thông qua các lệnh `make` đã được cấu hình sẵn:

### Chạy Pipeline Huấn luyện và Triển khai
Mở terminal và chạy lần lượt hoặc sử dụng `make all` để chạy toàn bộ:
```bash
# 1. Feature Engineering (Xử lý đặc trưng và lưu lên Feature Store)
make feature-engineering

# 2. Huấn luyện Retrieval Model (Mô hình Two-Tower)
make train-retrieval

# 3. Huấn luyện Ranking Model (Mô hình CatBoost)
make train-ranking

# 4. Tính toán và lưu trữ Vector Embeddings
make create-embeddings

# 5. Khởi tạo Deployments trên Hopsworks
make create-deployments

# 6. Lên lịch tự động hóa Materialization Jobs
make schedule-materialization-jobs
```

### Khởi chạy Giao diện UI (Streamlit)
Sau khi các deployments trên Hopsworks đã chạy thành công, bạn có thể bật UI:

**Dùng mô hình xếp hạng CatBoost (Mặc định):**
```bash
make start-ui
```
**Dùng mô hình xếp hạng LLM (OpenAI):**
```bash
make start-ui-llm-ranking
```

## Phần phụ thuộc (Dependencies)

Các thư viện chính được sử dụng trong dự án:
- `hopsworks[python] >= 4.1.2`: Quản lý Feature Store và Moddel Registry.
- `tensorflow-recommenders == 0.7.2` & `tensorflow == 2.14`: Xây dựng Retrieval model (Two-Tower).
- `catboost == 1.2`: Xây dựng Ranking model.
- `streamlit == 1.28.2`: Xây dựng giao diện web.
- `langchain` & `langchain-openai`: Tích hợp sức mạnh của LLM vào Ranking.
- `sentence-transformers == 2.2.2`: Tính toán Embedding Text.
- `polars`: Xử lý dữ liệu bảng với hiệu suất cao.
- **Trình quản lý gói**: `uv`

Xem danh sách đầy đủ tại `pyproject.toml`.

## Lỗi đã biết (Known Issues)

1. **Khởi động Deployment chậm:** Lần đầu gọi mô hình qua API của Hopsworks có thể mất một chút thời gian. Khi đó UI trên Streamlit có thể phản hồi chậm.
2. **Hết bộ nhớ (Out-of-Memory / OOM):** Quá trình huấn luyện (`train-retrieval`) có thể ngốn RAM nếu bạn chạy trên máy tính cá nhân. Có thể chỉnh `CUSTOMER_DATA_SIZE = CustomerDatasetSize.SMALL` trong file `recsys/config.py` để giảm tải.
3. **Giới hạn API:** Tính năng LLM Ranking sử dụng OpenAI API, có thể gặp lỗi Rate Limit (429) đối với các tài khoản miễn phí.

## Giải pháp khắc phục lỗi phổ biến (Troubleshooting)

**Hỏi: Không thể xác thực với Hopsworks (Unauthorized)?**
> **Đáp:** Hãy kiểm tra lại `HOPSWORKS_API_KEY` trong file `.env`. Đảm bảo API Key của bạn chưa hết hạn và có đầy đủ quyền truy cập Project.

**Hỏi: Làm sao để dọn dẹp tài nguyên để không bị trừ chi phí trên Hopsworks?**
> **Đáp:** Bạn nên tắt các Inference Deployment khi không dùng bằng lệnh:
> ```bash
> make clean-hopsworks-resources
> ```

## Tài liệu tham khảo (References)

Trong quá trình học tập và làm dự án, mình đã tham khảo các tài liệu sau:
- [Tài liệu chính thức của Hopsworks](https://docs.hopsworks.ai/)
- [TensorFlow Recommenders Tutorial](https://www.tensorflow.org/recommenders)
- [Tài liệu Streamlit](https://docs.streamlit.io/)
- [Kiến trúc Machine Learning Pipeline cho Recommender System.](https://cloud.google.com/architecture/machine-learning-on-gcp)

## Thông tin liên hệ (Contact)

Dự án này là sản phẩm cá nhân phục vụ môn học và nghiên cứu thực tế. 
Nếu bạn thấy dự án hữu ích hoặc có góp ý để mình cải thiện code/kiến trúc, vui lòng mở một Issue hoặc liên hệ qua email sinh viên của mình: 
- **Tên:** [Điền tên của bạn]
- **Email/LinkedIn:** [Điền email hoặc link LinkedIn]
- **Github:** [Link Github cá nhân]
