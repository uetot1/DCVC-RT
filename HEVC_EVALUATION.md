# HEVC x265 anchor for BPP-mAP / BD-rate

The HEVC anchor and DCVC-RT candidate must use the identical evaluation
manifest, frame count, YOLO weights, detector size, confidence/NMS thresholds,
and all-frame protocol.

## RGB/PNG source

For an RGB frame manifest, this evaluator uses a BT.709 full-range YUV444
10-bit interchange. It encodes the source with `x265`, decodes the actual
HEVC elementary stream, converts the reconstruction to RGB, and runs the
frozen YOLO detector on every frame.

`x265` is intentionally constrained to Low-Delay P:

- one first I-frame;
- no B-frames;
- fixed GOP equal to the evaluated sequence length;
- no scenecut intra refresh.

This is a reproducible traditional-video anchor compatible with the causal
DCVC-RT prediction structure. BPP counts the complete independently decodable
HEVC bitstream, including the first I-frame and headers.

```bash
python evaluate_hevc.py \
  --data-dir /data/vcm_eval \
  --dataset-manifest manifest.json \
  --x265-encoder x265 \
  --configuration-name "x265 HEVC Low-Delay P RGB444 10-bit" \
  --preset medium \
  --chroma-format 444 \
  --bit-depth 10 \
  --qps 22 27 32 37 \
  --yolov5-repo /opt/yolov5 \
  --yolov5-weights /weights/yolov5s.pt \
  --method-name hevc_x265_ldp_rgb444_10bit
```

The command needs an x265 build supporting the requested chroma format and
bit depth (the evaluator requests `main444-10` by default). Check it before
the long run:

```bash
x265 --version
```

Use `--resume` after an interruption. The script checkpoints after every
completed sequence, so already-completed sequences and QPs are not repeated.

## Source YUV420

Use `--chroma-format 420 --bit-depth 8` only when the original evaluation
source is truly YUV420. Do not mix a RGB444 result and a YUV420 result in one
BD-rate calculation.

## Four or more QPs and comparison

The QP values do not have to equal the DCVC-RT base QPs. They only need to
produce at least four Pareto rate-mAP points with a meaningful mAP overlap
with the candidate. Dense exploratory runs are supported, for example
`--qps 29 33 37 41 45 50`. All Pareto-optimal points are used by the BD-rate
calculation; the HEVC and candidate curves do not need the same point count.

Changing the QP list changes the evaluation identity. Use a new
`--progress-checkpoint` for a new list, and only add `--resume` when the QPs,
dataset, x265 configuration, and detector configuration exactly match the
saved progress file.

```bash
python evaluate_vcm.py --mode bdrate \
  --anchor-results output/hevc_evaluation/hevc_x265_ldp_rgb444_10bit_results.json \
  --candidate-results output/evaluation/dcvc_rt_vcm_results.json \
  --rate actual_bpp \
  --metric map5095 \
  --output-dir output/comparison_x265
```

Negative BD-rate means the DCVC-RT candidate uses fewer bits than x265 at the
same mAP. The checker rejects results with different data, detector settings,
or evaluated-frame counts.
