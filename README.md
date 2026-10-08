# EDAPIBench-Construction

Chạy các lệnh pipeline tại thư mục gốc repository. Cấu hình thử hiện tại: **NumPy, 10 mẫu completion đầu vào và DeepSeek-Coder 1.3B Base**.

## Chuẩn bị môi trường

Tạo và kích hoạt môi trường Python 3.11, sau đó cài các thư viện được sử dụng trong pipeline:

```bash
conda create -n EDAPIBench python=3.11 -y
conda activate EDAPIBench
python -m pip install torch transformers peft requests urllib3 tqdm tiktoken fuzzywuzzy huggingface_hub numpy openai datasets boto3 "smart_open[s3]"
```

Bước inference dùng CUDA, vì vậy bản PyTorch cần phù hợp với GPU/driver trên server. Danh sách trên chưa pin phiên bản dependency.

## Đặt token trong terminal

Thay các giá trị minh họa bằng token của bạn và chạy trong terminal sẽ dùng để chạy `search.py`.

Trên Linux/server hoặc Bash:

```bash
export SG_TOKEN="sgp_xxxxxxxxx"
export GITHUB_TOKEN="github_pat_xxxxxxxxx"
```

Hoặc trên Windows PowerShell:

```powershell
$env:SG_TOKEN = "sgp_xxxxxxxxx"
$env:GITHUB_TOKEN = "github_pat_xxxxxxxxx"
```

Hai biến này chỉ áp dụng cho phiên terminal hiện tại và các tiến trình được khởi chạy từ đó.

## Step 1: Tìm và tải code chứa API up-to-date

Dùng mapping **API deprecated → API thay thế** trong `data/mappings/` để tìm code trên GitHub qua Sourcegraph và GitHub Code Search, rồi tải các file source.

```bash
python code-search/search.py
```

Đầu ra: `data/searching-results/numpy/`.

## Step 2: Phân tích AST và trích các đoạn code

Phân tích AST, phân giải tên và alias để xác định lời gọi API; trích function hoặc đoạn code phù hợp và loại trùng.

```bash
python code-search/match.py
```

Đầu ra: `data/matching-results/numpy/matched-functions.json`.

## Step 3: Chuẩn hóa thành mẫu completion

Tạo `probing input` từ code trước dòng gọi API và `reference` từ dòng gọi API; bổ sung mapping và metadata. Cấu hình hiện tại lấy tối đa **10 mẫu**.

```bash
python code-search/standardize.py
```

Đầu ra: `data/standardized-results/numpy/standardized_samples.json`.

Khi chuyển sang server, cần chuyển cả file này vì `data/standardized-results/` nằm trong `.gitignore`.

## Step 4: Cho LLM sinh completion và lọc mẫu deprecated API

Model được tải tự động từ Hugging Face Hub ở lần chạy đầu và lưu vào cache, các lần sau dùng lại cache. Server cần có kết nối mạng và không đặt `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`.

Chạy script dưới đây trên Bash. Script gọi `dapi_inference.py` để DeepSeek sinh completion, rồi gọi `predicted_dapi_collection.py` để giữ mẫu sinh API deprecated tương ứng với mapping và loại trùng prompt.

```bash
bash dapi-collection.sh
```

Đầu ra: `data/predicted-dapi-results/numpy/deepseek-1.3b/data.json`. Số mẫu giữ lại có thể từ 0 đến 10.

## Step 5: Xây dữ liệu đánh giá

Từ các mẫu đã giữ, xây dữ liệu Generalization, Specificity, Portability và gộp thành EDAPIBench.

**Bước này vẫn dùng cấu hình gốc:** cần chỉnh phạm vi thư viện/model cho NumPy và DeepSeek, đồng thời cấu hình dịch vụ LLM tạo biến thể và quyền truy cập dữ liệu The Stack v2/S3 trước khi chạy.

```bash
bash eval-data-construction.sh
```

- **Effectiveness:** dùng chính các mẫu gốc giữ lại ở Step 4.
- **Generalization:** tạo biến thể code nhằm giữ nguyên ngữ nghĩa, rồi dùng model kiểm tra và chọn ứng viên.
- **Specificity:** thu thập code ngoài mục tiêu chỉnh sửa và lưu prediction của model gốc để kiểm tra việc giữ nguyên hành vi.
- **Portability:** liên kết với một mẫu khác mà model sinh cùng API deprecated để kiểm tra trên ngữ cảnh khác.

Đầu ra tổng quát: `data/EDAPIBench/<model>/all.json`.

Pipeline này xây dữ liệu benchmark; chưa thực hiện chỉnh sửa model hay tính điểm đánh giá sau chỉnh sửa.
