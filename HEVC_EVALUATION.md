# HEVC HM anchor cho BPP–mAP/BD-rate

Anchor phải dùng đúng cùng sequence, frame count, ground truth, YOLO weights,
input size, confidence/NMS thresholds và **all-frame protocol** với candidate.

## Source PNG/RGB

Theo test condition của Microsoft DCVC, traditional codec nên dùng YUV444 10-bit
khi source là RGB. Cần HM RExt configuration, ví dụ
`encoder_lowdelay_main_rext.cfg`:

```bash
python evaluate_hevc.py \
  --data-dir /data/vcm_eval \
  --dataset-manifest manifest.json \
  --hm-encoder /opt/HM/bin/TAppEncoderStatic \
  --hm-config /opt/HM/cfg/encoder_lowdelay_main_rext.cfg \
  --configuration-name "HM Low-Delay RGB444 10-bit" \
  --protocol all-frames \
  --chroma-format 444 \
  --bit-depth 10 \
  --qps 22 27 32 37 \
  --hm-extra-arg=--IntraPeriod=-1 \
  --yolov5-repo /opt/yolov5 \
  --yolov5-weights /weights/yolov5s.pt \
  --method-name hevc_hm_rgb444
```

Pipeline:

```text
RGB PNG → BT.709 full-range YUV444 10-bit → HM → YUV444 → RGB → YOLO
```

HM bitstream được tính toàn bộ, bao gồm header và I-frame. mAP được tính trên
mọi frame.

## Source YUV420 thật

Nếu test sequence gốc là YUV420, thiết lập chuẩn nhất là mã hóa trực tiếp file
YUV420 gốc. Script hiện nhận frame manifest PNG để đồng bộ ground truth; chế độ
`--chroma-format 420` chỉ hợp lệ khi các PNG được xác định rõ là representation
của cùng source protocol. Không trộn kết quả RGB444 và YUV420 trong một phép
BD-rate.

```bash
--chroma-format 420 --bit-depth 8
```

Pipeline này dùng BT.709 RGB full ↔ YUV420 limited. Width/height phải chẵn.

## Chọn QP

HEVC không cần dùng cùng số QP với DCVC-RT; cần bốn rate points có vùng mAP giao
nhau với candidate. Nếu `22/27/32/37` không overlap, chạy pilot rồi dịch cả dải.
Không loại điểm dominated rồi vẫn báo BD-rate: bốn điểm đưa vào phải tạo Pareto
front với bitrate và mAP tăng cùng nhau.

## So sánh ba codec

```bash
python compare_codecs_bd_rate.py \
  --hevc output/hevc_evaluation/hevc_hm_rgb444_results.json \
  --learned-scalable results/learned_scalable_results.json \
  --proposed output/evaluation/dcvc_rt_vcm_results.json \
  --rate actual_bpp \
  --metrics map50 map5095 \
  --output-dir output/comparison
```

Script từ chối kết quả khác `evaluation_id`, detector config, ground truth,
protocol hoặc evaluated-frame count. Mỗi JSON bắt buộc khai báo riêng
`machine_frontend`: HEVC/DCVC-RT anchor dùng pretrained front end, còn candidate
dùng cloned front end đã train. Đây là so sánh hệ thống VCM end-to-end, không
phải cùng toàn bộ detector weights. BD-rate âm nghĩa là candidate dùng ít bit
hơn anchor tại cùng mAP.
