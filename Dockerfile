# BEAVER — inference container for the REG2026 challenge.
#
# The base image provides the WSI feature-extraction stack (PyTorch + trident +
# openslide + timm). It is not public; build it first with base.Dockerfile:
#   docker build -f base.Dockerfile -t feature-extraction:v1 .
FROM --platform=linux/amd64 feature-extraction:v1 AS beaver_amd64

ENV PYTHONUNBUFFERED=1
# entry imports: `from core import ...` and `src.interf*`
ENV PYTHONPATH=/opt/app
# offline weights: HF cache lives under the mounted model dir (/opt/ml/model)
ENV HF_HOME=/opt/ml/model/hoptimus1_hf
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# --- retrieval / report routing ---
ENV COT_REPORT_ROUTING=1
ENV COT_ROUTE_ORGANS=prostate,bladder,cervix,lung
ENV COT_PATCH_CAP=1000
ENV COT_HOPT_TIME_BUDGET=180

# --- per-slide robustness (avoid submission-wide timeout on slow/atypical WSIs) ---
#   COT_SLIDE_DEADLINE : interf1 kills the inference subprocess after this many seconds
#                        and returns a fallback CoT, so a single hanging slide cannot
#                        exceed the platform time limit.
#   COT_SEG_TIME_BUDGET: wall-clock guard for the segmentation stage.
ENV COT_SLIDE_DEADLINE=300
ENV COT_SEG_TIME_BUDGET=90
#   Single-layer giant-tile slides (no pyramid): stride-sample tiles for segmentation
#   (<= COT_SEG_MAX_TILES) and cap native full-decode thumbnails.
ENV COT_SEG_MAX_TILES=6000
ENV COT_SINGLE_NATIVE_MAX=50000000

# --- content-based artifact correction (no effect on normal slides) ---
#   COT_FIX_YCBCR      : detect pink-cast (mis-tagged JPEG YCbCr) -> YCbCr->RGB correction
#   COT_REMOVE_PENMARKS: exclude high-saturation pure-color (pen ink) from segmentation
ENV COT_FIX_YCBCR=1
ENV COT_REMOVE_PENMARKS=1

# --- agentic self-verification + Metric B tissue/background gate ---
#   COT_SELF_VERIFY: #1-diagnosis-anchored slot consistency check/repair with a reasoning trace.
#   COT_B2_SAT_THR : hybrid (encoder OR saturation) tissue gate threshold.
ENV COT_SELF_VERIFY=1
ENV COT_B2_SAT_THR=0.05

# --- counterfactual reasoning (Metric C); fully dormant unless a spec is injected ---
#   When no counterfactual spec is present, output is byte-identical to normal inference.
#   When a spec is injected, attribute-constrained retrieval re-derives the conclusion
#   and downstream chain. See pipeline/counterfactual.py.
ENV COT_COUNTERFACTUAL=1

# --- A7 attribute-accuracy levers (held-out verified; LLM-free, deterministic) ---
#   COT_ART           : artifact dir with the defect-fixed retrieval DB and knowledge bases.
#   COT_INV_BRANCH    : per-branch invasion resolved from the branch diagnosis (bladder/breast).
#   COT_CONSISTENCY   : per-branch papillary + behavior resolution (behavior is purity-gated).
#   COT_DXFIX=0       : keep #1 diagnosis from the slot (do NOT unify it to the retrieved report).
#   COT_GRADE_OVERRIDE: replace the routed prostate report grade with the Gleason-score slot.
ENV COT_ART=/opt/ml/model/cot_artifacts
ENV COT_INV_BRANCH=1
ENV COT_CONSISTENCY=1
ENV COT_DXFIX=0
ENV COT_GRADE_OVERRIDE=1

WORKDIR /opt/app

# extra deps not in base (hydra / omegaconf / scikit-learn / joblib)
COPY requirements.txt /opt/app/
RUN python -m pip install --no-cache-dir --no-color --requirement /opt/app/requirements.txt

# pre-seed torchvision deeplabv3 resnet50 (ImageNet) weights so no download at runtime
ENV TORCH_HOME=/opt/torch_cache
COPY image_cache/torch/ /opt/torch_cache/

# point the trident registry at the offline tissue-segmentation checkpoint
RUN python -c "import trident, os, json; \
p = os.path.join(os.path.dirname(trident.__file__), 'segmentation_models', 'local_ckpts.json'); \
d = json.load(open(p)); d['hest'] = '/opt/ml/model/trident_seg/deeplabv3_seg_v4.ckpt'; \
json.dump(d, open(p, 'w')); print('[seg registry]', d)"

# code only (weights are mounted at runtime from /opt/ml/model)
COPY core.py      /opt/app/
COPY inference.py /opt/app/
COPY src/         /opt/app/src/
# model_lib: model definition (imported as `src.*` in a separate process); pipeline: CoT assembly
COPY model_lib/   /opt/app/model_lib/
COPY pipeline/    /opt/app/pipeline/

# Grand Challenge rejects containers running as root -> switch to a non-root user.
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

ENTRYPOINT ["python", "inference.py"]
