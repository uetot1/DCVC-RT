# Rebuild notes from demo commit 5c12c82

Các thay đổi quyết định so với demo ban đầu:

1. Thêm DCVC-RT `DMCI`; nạp image checkpoint, đóng băng và dùng reconstructed
   I-frame làm reference cho DMC.
2. Sửa toàn bộ color path thành RGB ↔ full-range BT.709 YCbCr 4:4:4; YOLO không
   nhận nhầm YCbCr/RGB.
3. Evaluation chuyển từ external-seed P-frame-only sang all-frame actual
   bitstream: tính cả bit và mAP frame 0.
4. Giữ QP offset `0,8,0,4,...` vì khớp `index_map`/`shift_qp` của DCVC-RT; bỏ
   frame-distortion weights không có căn cứ trong implementation chính thức.
5. Thêm `lambda_scale`; dải `1→768` chỉ cung cấp hình dạng QP–λ, không được coi
   là λ feature đã được paper xác nhận.
6. Thêm DDP, DistributedSampler, `no_sync` gradient accumulation, per-rank random
   QP và RNG state trong checkpoint.
7. Curriculum mặc định đổi thành `3→5→7` tại epoch `1,3,6`, phù hợp run 15 epoch.
8. LR fine-tune đổi thành DMC `1e-5`, cloned front end `1e-6`; scheduler tại
   epoch `8,12`.
9. Log thêm RGB PSNR/MSE, Y/chroma MSE, I-frame PSNR, gradient norm, active frame
   count, GPU memory/time và validation riêng bốn QP.
10. `best.pt` raw-loss cũ được thay bằng `best_proxy.pt` dựa trên geometric mean
    của bốn objectives. Checkpoint final vẫn phải chọn bằng actual BPP–mAP.
11. HEVC RGB anchor mặc định đổi sang YUV444 10-bit/all-frame; YUV420 trở thành
    lựa chọn riêng cho protocol có source YUV420.
12. Thêm dataset validator, protocol tests và tài liệu Kaggle mới.
13. Rate surrogate Gaussian dùng đúng scale table entropy coder `[0.11,16]`
    với 128 mức log và straight-through gradient.
14. Training luôn dùng đường PyTorch có autograd; fused CUDA extension chỉ được
    phép chạy khi global grad đã tắt.
15. Evaluator khớp DCVC-RT official inference: FP16, reset interval 32 và hai
    entropy coder cho nội dung lớn hơn `1280×720`; container VCM2 tự lưu reset.
16. Kết quả actual-bitstream ghi actual/estimated BPP gap và fingerprint cloned
    front end, xác định rõ đây là so sánh hệ thống VCM end-to-end.
17. Main feature loss chuyển từ backbone `4/6/9` sang đúng ba PAN/FPN tensor
    `17/20/23` được đưa trực tiếp vào YOLOv5 Detect; `4/6/9` chỉ còn là ablation.
18. Cloned CV front end được khóa đúng năm layer `0..4`; task back end `5..23`
    đóng băng. Checkpoint nâng lên schema 12 để không resume nhầm topology cũ.
19. DDP tự bật `find_unused_parameters` khi TBPTT được dùng; evaluator bổ sung
    BPP/mAP theo temporal bins cho long-sequence drift test từ 100 frame.

Giới hạn còn lại mang tính khoa học, không phải lỗi code: DCVC-RT không công bố
training source riêng; `lambda_scale` và feature-level weights cần sweep/ablation
trên dữ liệu thực trước khi chốt kết quả paper.
