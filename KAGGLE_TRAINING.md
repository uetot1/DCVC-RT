# Kaggle: train DCVC-RT VCM trên 2 GPU T4

Đây là checklist vận hành cho bản final. Giải thích protocol và căn cứ khoa học
nằm trong `README.md`.

## 1. Input datasets

```text
/kaggle/input/vimeo90k/vimeo_septuplet/
├── sequences/<id1>/<id2>/im1.png ... im7.png
├── sep_trainlist.txt
└── sep_testlist.txt

/kaggle/input/dcvc-rt-weights/
├── cvpr2025_image.pth.tar
└── cvpr2025_video.pth.tar

/kaggle/input/yolov5-v7-offline/
├── yolov5/                 # checkout đúng tag v7.0
└── yolov5s.pt
```

Không đưa checkpoint vào GitHub ZIP.

## 2. Clone/cài đặt

```bash
%cd /kaggle/working
!unzip -q /kaggle/input/dcvc-rt-vcm-source/DCVC-RT-VCM-final.zip
%cd /kaggle/working/DCVC-RT-VCM-final
!pip install -q tqdm scipy matplotlib pybind11 opencv-python-headless
```

Không cài lại `torch`/`torchvision` nếu Kaggle đã nhận đủ 2 GPU.

```python
import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
```

## 3. Khai báo đường dẫn một lần

```python
from pathlib import Path

VIMEO = Path("/kaggle/input/vimeo90k/vimeo_septuplet")
IMAGE_CKPT = Path("/kaggle/input/dcvc-rt-weights/cvpr2025_image.pth.tar")
VIDEO_CKPT = Path("/kaggle/input/dcvc-rt-weights/cvpr2025_video.pth.tar")
YOLO_REPO = Path("/kaggle/input/yolov5-v7-offline/yolov5")
YOLO_WEIGHTS = Path("/kaggle/input/yolov5-v7-offline/yolov5s.pt")

for path in (
    VIMEO / "sequences",
    VIMEO / "sep_trainlist.txt",
    VIMEO / "sep_testlist.txt",
    IMAGE_CKPT,
    VIDEO_CKPT,
    YOLO_REPO,
    YOLO_WEIGHTS,
):
    assert path.exists(), path
```

## 4. Validate dataset

```bash
!python validate_dataset.py \
  --root "$VIMEO/sequences" \
  --list-file "$VIMEO/sep_trainlist.txt" \
  --frames 7 --crop-size 256 --training
```

Nếu biến Python không được shell cell nhận, thay bằng đường dẫn đầy đủ. Chỉ tiếp
tục khi output là `status: PASS`.

## 5. DDP smoke test

```bash
!torchrun --standalone --nproc_per_node=2 train_vcm_final.py \
  --training-stage vimeo7 \
  --data-dir /kaggle/input/vimeo90k/vimeo_septuplet/sequences \
  --train-list /kaggle/input/vimeo90k/vimeo_septuplet/sep_trainlist.txt \
  --val-dir /kaggle/input/vimeo90k/vimeo_septuplet/sequences \
  --val-list /kaggle/input/vimeo90k/vimeo_septuplet/sep_testlist.txt \
  --image-checkpoint /kaggle/input/dcvc-rt-weights/cvpr2025_image.pth.tar \
  --video-init /kaggle/input/dcvc-rt-weights/cvpr2025_video.pth.tar \
  --yolov5-repo /kaggle/input/yolov5-v7-offline/yolov5 \
  --yolov5-weights /kaggle/input/yolov5-v7-offline/yolov5s.pt \
  --checkpoint-dir /kaggle/working/checkpoints/smoke \
  --crop-size 128 \
  --epochs 1 \
  --max-batches 4 \
  --max-validation-batches 2 \
  --validate-every 1 \
  --save-every 0
```

Kiểm tra:

```bash
!find /kaggle/working/checkpoints/smoke -maxdepth 2 -type f -printf '%p\n'
!nvidia-smi
```

Phải có `world_size=2`, `skipped=0`, `latest.pt`, `run_config.json`, CSV và
JSONL. Training script tự khóa fused CUDA inference và in
`autograd-safe PyTorch path`; đây là hành vi bắt buộc để gradient đúng.

## 6. Stress test 7 frame / crop 256

Chạy cấu hình nặng nhất trước full run:

```bash
!torchrun --standalone --nproc_per_node=2 train_vcm_final.py \
  --training-stage vimeo7 \
  --data-dir /kaggle/input/vimeo90k/vimeo_septuplet/sequences \
  --train-list /kaggle/input/vimeo90k/vimeo_septuplet/sep_trainlist.txt \
  --image-checkpoint /kaggle/input/dcvc-rt-weights/cvpr2025_image.pth.tar \
  --video-init /kaggle/input/dcvc-rt-weights/cvpr2025_video.pth.tar \
  --yolov5-repo /kaggle/input/yolov5-v7-offline/yolov5 \
  --yolov5-weights /kaggle/input/yolov5-v7-offline/yolov5s.pt \
  --checkpoint-dir /kaggle/working/checkpoints/stress_7f_256 \
  --crop-size 256 --epochs 1 \
  --vimeo-curriculum-frames 7 \
  --vimeo-curriculum-start-epochs 1 \
  --accumulation-steps 1 --tbptt-steps 0 \
  --max-batches 2 --validate-every 99 --save-every 0
```

Chạy thêm test nhánh TBPTT (kể cả khi full-BPTT không OOM) để xác nhận DDP xử lý
đúng parameter không dùng ở từng chunk:

```bash
!torchrun --standalone --nproc_per_node=2 train_vcm_final.py \
  --training-stage vimeo7 \
  --data-dir /kaggle/input/vimeo90k/vimeo_septuplet/sequences \
  --train-list /kaggle/input/vimeo90k/vimeo_septuplet/sep_trainlist.txt \
  --image-checkpoint /kaggle/input/dcvc-rt-weights/cvpr2025_image.pth.tar \
  --video-init /kaggle/input/dcvc-rt-weights/cvpr2025_video.pth.tar \
  --yolov5-repo /kaggle/input/yolov5-v7-offline/yolov5 \
  --yolov5-weights /kaggle/input/yolov5-v7-offline/yolov5s.pt \
  --checkpoint-dir /kaggle/working/checkpoints/stress_tbptt_7f \
  --crop-size 128 --epochs 1 \
  --vimeo-curriculum-frames 7 \
  --vimeo-curriculum-start-epochs 1 \
  --accumulation-steps 1 --tbptt-steps 2 \
  --max-batches 3 --validate-every 99 --save-every 0
```

`run_config.json` phải ghi `ddp_find_unused_parameters: true`. Với main
`tbptt=0`, giá trị phải là `false`. Nếu 7-frame/crop-256 OOM, dùng
`--tbptt-steps 2` cho final run.

## 7. λ-scale pilot sweep

Chạy ba job với `--lambda-scale 0.003`, `0.006`, `0.010`, mỗi job tối thiểu vài
trăm batch (ví dụ `--max-batches 300`) và ghi vào checkpoint directory riêng.
Không chạy song song ba job trên cùng hai GPU. So sánh:

- `estimated_bpp`;
- `feature_mse`;
- `dmc_grad_norm` và `frontend_grad_norm`;
- validation tại QP `0/21/42/63`;
- không có non-finite/skipped batch.

`0.006` là mặc định khởi tạo, không phải giá trị được paper công bố. Kết quả
gradient cũ của layer `4/6/9` không đủ để chọn λ cho main topology `17/20/23`;
pilot phải chạy lại sau thay đổi này.

## 8. Final 15-epoch run

Ví dụ sau giả định sweep đã chọn `0.006`. Main topology mặc định là feature
layer `17 20 23` và cloned front end `0..4`; không cần truyền thêm cờ. Layer
`4 6 9` chỉ dùng cho ablation riêng.

```bash
!torchrun --standalone --nproc_per_node=2 train_vcm_final.py \
  --training-stage vimeo7 \
  --data-dir /kaggle/input/vimeo90k/vimeo_septuplet/sequences \
  --train-list /kaggle/input/vimeo90k/vimeo_septuplet/sep_trainlist.txt \
  --val-dir /kaggle/input/vimeo90k/vimeo_septuplet/sequences \
  --val-list /kaggle/input/vimeo90k/vimeo_septuplet/sep_testlist.txt \
  --image-checkpoint /kaggle/input/dcvc-rt-weights/cvpr2025_image.pth.tar \
  --video-init /kaggle/input/dcvc-rt-weights/cvpr2025_video.pth.tar \
  --yolov5-repo /kaggle/input/yolov5-v7-offline/yolov5 \
  --yolov5-weights /kaggle/input/yolov5-v7-offline/yolov5s.pt \
  --checkpoint-dir /kaggle/working/checkpoints/vcm_vimeo7 \
  --epochs 15 \
  --crop-size 256 \
  --batch-size-per-gpu 1 \
  --accumulation-steps 4 \
  --tbptt-steps 0 \
  --learning-rate 1e-5 \
  --frontend-learning-rate 1e-6 \
  --lambda-scale 0.006 \
  --vimeo-curriculum-frames 3 5 7 \
  --vimeo-curriculum-start-epochs 1 3 6 \
  --validation-qps 0 21 42 63 \
  --validate-every 1 \
  --max-validation-batches 25 \
  --save-every 5 \
  --keep-periodic-checkpoints 3
```

Nếu OOM tại epoch 6 khi bắt đầu bảy frame:

1. thử `--tbptt-steps 2`;
2. nếu vẫn OOM, dùng crop 128 cho thí nghiệm pilot;
3. không giảm số frame final xuống dưới 7 mà vẫn ghi là seven-frame training.

## 9. Resume đúng cách

```bash
!torchrun --standalone --nproc_per_node=2 train_vcm_final.py \
  --training-stage vimeo7 \
  --data-dir /kaggle/input/vimeo90k/vimeo_septuplet/sequences \
  --train-list /kaggle/input/vimeo90k/vimeo_septuplet/sep_trainlist.txt \
  --val-dir /kaggle/input/vimeo90k/vimeo_septuplet/sequences \
  --val-list /kaggle/input/vimeo90k/vimeo_septuplet/sep_testlist.txt \
  --image-checkpoint /kaggle/input/dcvc-rt-weights/cvpr2025_image.pth.tar \
  --resume /kaggle/input/previous-output/vcm_vimeo7/latest.pt \
  --yolov5-repo /kaggle/input/yolov5-v7-offline/yolov5 \
  --yolov5-weights /kaggle/input/yolov5-v7-offline/yolov5s.pt \
  --checkpoint-dir /kaggle/working/checkpoints/vcm_vimeo7 \
  --epochs 15
```

Không truyền `--video-init` cùng `--resume`. Resume cần giữ nguyên feature
layers, cloned-front-end topology, λ, curriculum, TBPTT và optimizer settings
của run cũ. Checkpoint schema cũ dùng clone `0..9` không được resume vào bản này.

## 10. Sau training

Giữ tối thiểu:

```text
latest.pt
best_proxy.pt
epoch_5.pt
epoch_10.pt
epoch_15.pt
run_config.json
logs/*.csv
logs/*_batches.jsonl
logs/*_training_curves.png
logs/latest_training_history.csv
logs/latest_training_curves.png
```

CSV được cập nhật sau mỗi epoch. PNG gồm ba đồ thị `total loss`, `estimated BPP`
và `Feature MSE`, được vẽ một lần khi run kết thúc hoặc cleanup sau `Ctrl+C`/
exception. Nếu resume từ checkpoint mới, CSV/PNG của run sau chứa lại cả lịch sử
epoch trước đó vì lịch sử được lưu bên trong checkpoint.

`best_proxy.pt` không tự động là checkpoint có BD-rate tốt nhất. Chạy actual
BPP–mAP trên validation có nhãn cho `epoch_5/10/15`, rồi mới chọn checkpoint
final. Sau đó Save Version/Save & Run All để output được giữ lại.

Actual-bitstream evaluation mặc định phải giữ `--reset-interval 32` và
`--codec-precision fp16`. Kết quả kiểm tra thêm
`actual_to_estimated_bpp_ratio`; nếu lệch lớn/bất thường ở một QP thì chưa dùng
đường đó để báo cáo BD-rate.

## 11. Long-sequence evaluation bắt buộc

Main training vẫn kết thúc sau 15 epoch trên clip bảy frame. Không train thêm
long sequence trước khi có bằng chứng drift. Sau training, chạy candidate và
DCVC-RT anchor trên cùng tập sequence có ít nhất 100 frame:

```bash
!python evaluate_vcm.py --mode codec \
  --data-dir /kaggle/input/vcm-long-eval \
  --dataset-manifest /kaggle/input/vcm-long-eval/manifest.json \
  --image-ckpt /kaggle/input/dcvc-rt-weights/cvpr2025_image.pth.tar \
  --video-ckpt /kaggle/working/checkpoints/vcm_vimeo7/epoch_15.pt \
  --yolov5-repo /kaggle/input/yolov5-v7-offline/yolov5 \
  --yolov5-weights /kaggle/input/yolov5-v7-offline/yolov5s.pt \
  --method-name dcvc_rt_vcm_epoch15 \
  --qps 0 21 42 63 \
  --reset-interval 32 \
  --codec-precision fp16 \
  --minimum-sequence-frames 100
```

JSON tự ghi BPP và mAP theo `frame_0`, `frames_1_7`, `frames_8_31`,
`frames_32_63`, `frames_64_plus`. Chỉ fine-tune long sequence 1–3 epoch với LR
thấp/TBPTT nếu candidate có BPP tăng hoặc mAP giảm theo frame index rõ hơn
pretrained DCVC-RT anchor.
