"""YOLOv5 task-pyramid extractors used by the machine VCM objective."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn


# YOLOv5 v7.0 Detect consumes the three PAN/FPN tensors produced by these
# layers. This is the main TransTIC-inspired task-pyramid objective.
DEFAULT_FEATURE_LAYER_INDICES = (17, 20, 23)

# Retained only as an explicitly named ablation. These are backbone stages, not
# the tensors directly consumed by Detect.
BACKBONE_ABLATION_FEATURE_LAYER_INDICES = (4, 6, 9)

# Learned Scalable Video Coding clones the first five YOLOv5 layers. The
# original task back end (layers 5..23 and Detect) remains frozen.
DEFAULT_CLONED_FRONTEND_LAST_LAYER = 4
LAST_SUPPORTED_FEATURE_LAYER = 23


@contextmanager
def _allow_legacy_yolov5_checkpoint_loading():
    """Temporarily support YOLOv5 v7 checkpoints with PyTorch >= 2.6."""
    original_load = torch.load

    def compatible_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = compatible_load
    try:
        yield
    finally:
        torch.load = original_load


def load_yolov5(
    model_name: str = "yolov5s",
    repository: str | Path | None = None,
    weights: str | Path | None = None,
):
    """Load pinned YOLOv5 v7 online or from Kaggle/local files."""
    repository_path = Path(repository).expanduser() if repository else None
    weights_path = Path(weights).expanduser() if weights else None
    if repository_path is not None and not repository_path.is_dir():
        raise FileNotFoundError(f"YOLOv5 repository not found: {repository_path}")
    if weights_path is not None and not weights_path.is_file():
        raise FileNotFoundError(f"YOLOv5 weights not found: {weights_path}")

    source = "local" if repository_path is not None else "github"
    repo_or_dir = (
        str(repository_path)
        if repository_path is not None
        else "ultralytics/yolov5:v7.0"
    )
    entrypoint = "custom" if weights_path is not None else model_name
    load_kwargs = {
        "source": source,
        "trust_repo": True,
    }
    if weights_path is not None:
        load_kwargs["path"] = str(weights_path)
    else:
        load_kwargs["pretrained"] = True

    with _allow_legacy_yolov5_checkpoint_loading():
        return torch.hub.load(
            repo_or_dir,
            entrypoint,
            **load_kwargs,
        )


def ddp_find_unused_parameters(tbptt_steps: int) -> bool:
    """Return whether a TBPTT DDP forward can omit DMC parameters.

    A full-GOP forward uses both ``feature_adaptor_i`` and
    ``feature_adaptor_p``. With TBPTT, later chunks no longer use the I-frame
    adaptor, so DDP must discover unused parameters per forward.
    """
    if int(tbptt_steps) < 0:
        raise ValueError("tbptt_steps must be non-negative")
    return int(tbptt_steps) > 0


def _layer_source(layer) -> int | list[int]:
    source = getattr(layer, "f", -1)
    if isinstance(source, int):
        return int(source)
    return [int(index) for index in source]


def _run_yolov5_graph(
    layers: nn.ModuleList,
    images: torch.Tensor,
    saved_layer_indices: frozenset[int],
    selected_layer_indices: frozenset[int],
) -> dict[int, torch.Tensor]:
    """Run a YOLOv5 graph prefix with the same skip routing as `_forward_once`."""
    outputs: list[torch.Tensor | None] = []
    selected: dict[int, torch.Tensor] = {}
    current: torch.Tensor | list[torch.Tensor] = images
    for fallback_index, layer in enumerate(layers):
        layer_index = int(getattr(layer, "i", fallback_index))
        source = _layer_source(layer)
        if source != -1:
            if isinstance(source, int):
                routed = outputs[source]
                if routed is None:
                    raise RuntimeError(
                        f"YOLOv5 layer {layer_index} requires unsaved layer {source}"
                    )
                current = routed
            else:
                routed_inputs = []
                for source_index in source:
                    if source_index == -1:
                        routed_inputs.append(current)
                    else:
                        routed = outputs[source_index]
                        if routed is None:
                            raise RuntimeError(
                                f"YOLOv5 layer {layer_index} requires unsaved "
                                f"layer {source_index}"
                            )
                        routed_inputs.append(routed)
                current = routed_inputs
        current = layer(current)
        if layer_index in selected_layer_indices:
            selected[layer_index] = current
        outputs.append(current if layer_index in saved_layer_indices else None)
    return selected


class YOLOv5FeatureExtractor(nn.Module):
    """Extract YOLOv5 task-pyramid features with a five-layer cloned front end.

    The main objective uses layers 17/20/23, the three tensors passed directly
    to Detect. Only layers 0..4 of the reconstruction path can be optimized;
    layers 5..23 are a frozen copy of the pretrained task back end. The teacher
    is frozen end to end. BatchNorm statistics stay fixed on both paths.
    """

    def __init__(
        self,
        model_name: str = "yolov5s",
        feature_layer_indices: Sequence[int] = DEFAULT_FEATURE_LAYER_INDICES,
        repository: str | Path | None = None,
        weights: str | Path | None = None,
        trainable: bool = False,
        cloned_frontend_last_layer: int = DEFAULT_CLONED_FRONTEND_LAST_LAYER,
    ):
        super().__init__()
        indices = tuple(int(index) for index in feature_layer_indices)
        if not indices:
            raise ValueError("feature_layer_indices must contain at least one layer")
        if any(index < 0 for index in indices):
            raise ValueError("feature layer indices must be non-negative")
        if len(set(indices)) != len(indices):
            raise ValueError("feature layer indices must be unique")
        if tuple(sorted(indices)) != indices:
            raise ValueError("feature layer indices must be in ascending order")
        if indices[-1] > LAST_SUPPORTED_FEATURE_LAYER:
            raise ValueError(
                "feature layers are limited to YOLOv5 layers 0.."
                f"{LAST_SUPPORTED_FEATURE_LAYER}; layer 24 is Detect output"
            )

        cloned_frontend_last_layer = int(cloned_frontend_last_layer)
        if cloned_frontend_last_layer != DEFAULT_CLONED_FRONTEND_LAST_LAYER:
            raise ValueError(
                "cloned_frontend_last_layer must be 4 to preserve the exact "
                "five-layer Learned Scalable front-end protocol"
            )

        yolo = load_yolov5(model_name, repository=repository, weights=weights)
        detection_model = yolo.model.model
        layers = list(detection_model.model.children())
        final_layer = max(indices[-1], cloned_frontend_last_layer)
        if final_layer >= len(layers):
            raise ValueError(
                f"YOLOv5 has only {len(layers)} layers, but layer {final_layer} "
                "was requested"
            )

        self.feature_layer_indices = indices
        self._selected_layer_indices = frozenset(indices)
        self.cloned_frontend_last_layer = cloned_frontend_last_layer
        self.graph_prefix = nn.ModuleList(layers[: final_layer + 1])
        official_save = {
            int(index)
            for index in getattr(detection_model, "save", ())
            if int(index) <= final_layer
        }
        self._saved_layer_indices = frozenset(
            official_save | set(indices) | {final_layer}
        )
        self.set_trainable(trainable)

    def _cloned_frontend(self) -> nn.ModuleList:
        return nn.ModuleList(
            list(self.graph_prefix[: self.cloned_frontend_last_layer + 1])
        )

    def cloned_frontend_state_dict(self) -> dict[str, torch.Tensor]:
        return self._cloned_frontend().state_dict()

    def load_cloned_frontend_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        self._cloned_frontend().load_state_dict(state_dict, strict=True)

    def cloned_frontend_named_parameters(self):
        if not self.trainable:
            return iter(())
        return self._cloned_frontend().named_parameters()

    def set_trainable(self, trainable: bool) -> None:
        """Train only cloned layers 0..4 while freezing the task back end."""
        self.trainable = bool(trainable)
        for layer_index, layer in enumerate(self.graph_prefix):
            requires_grad = self.trainable and (
                layer_index <= self.cloned_frontend_last_layer
            )
            for parameter in layer.parameters():
                parameter.requires_grad_(requires_grad)
        self.graph_prefix.eval()

    def train(self, mode: bool = True):
        """Keep BatchNorm and other stateful task layers fixed."""
        super().train(False)
        return self

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = _run_yolov5_graph(
            self.graph_prefix,
            images,
            self._saved_layer_indices,
            self._selected_layer_indices,
        )
        return tuple(features[index] for index in self.feature_layer_indices)


def install_cloned_frontend(
    detector,
    cloned_state_dict: dict[str, torch.Tensor],
    last_frontend_layer: int = DEFAULT_CLONED_FRONTEND_LAST_LAYER,
):
    """Install trained layers 0..4 into a full pretrained YOLOv5 detector."""
    last_frontend_layer = int(last_frontend_layer)
    if last_frontend_layer != DEFAULT_CLONED_FRONTEND_LAST_LAYER:
        raise ValueError(
            "last_frontend_layer must be 4 for the five-layer front-end protocol"
        )
    layers = list(detector.model.model.model.children())
    if last_frontend_layer >= len(layers):
        raise ValueError(
            f"YOLOv5 detector has only {len(layers)} layers, but cloned front "
            f"end ends at layer {last_frontend_layer}"
        )
    prefix = nn.ModuleList(layers[: last_frontend_layer + 1])
    prefix.load_state_dict(cloned_state_dict, strict=True)
    prefix.eval()
    for parameter in prefix.parameters():
        parameter.requires_grad_(False)
    return detector


# Backward-compatible import name for external scripts; new code and metadata
# use the scientifically accurate "front end" terminology.
install_cloned_backbone = install_cloned_frontend
