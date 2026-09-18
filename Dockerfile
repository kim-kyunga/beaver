FROM --platform=linux/amd64 feature-extraction:v1 AS beaver_amd64

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/app
ENV HF_HOME=/opt/ml/model/hoptimus1_hf
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

ENV COT_REPORT_ROUTING=1
ENV COT_ROUTE_ORGANS=prostate,bladder,cervix,lung
ENV COT_PATCH_CAP=1000
ENV COT_HOPT_TIME_BUDGET=180

ENV COT_SLIDE_DEADLINE=300
ENV COT_SEG_TIME_BUDGET=90
ENV COT_SEG_MAX_TILES=6000
ENV COT_SINGLE_NATIVE_MAX=50000000

ENV COT_FIX_YCBCR=1
ENV COT_REMOVE_PENMARKS=1

ENV COT_SELF_VERIFY=1
ENV COT_B2_SAT_THR=0.05

# Optional, OFF by default: confidence-gated report fix (requires dx1_heads.pkl under COT_ART).
# ENV COT_SV_REPORT=1

ENV COT_COUNTERFACTUAL=1

ENV COT_ART=/opt/ml/model/cot_artifacts
ENV COT_INV_BRANCH=1
ENV COT_CONSISTENCY=1
ENV COT_DXFIX=0
ENV COT_GRADE_OVERRIDE=1
ENV COT_CONF_GATE=1

WORKDIR /opt/app

COPY requirements.txt /opt/app/
RUN python -m pip install --no-cache-dir --no-color --requirement /opt/app/requirements.txt

RUN python -c "import trident, os, json; \
p = os.path.join(os.path.dirname(trident.__file__), 'segmentation_models', 'local_ckpts.json'); \
d = json.load(open(p)); d['hest'] = '/opt/ml/model/trident_seg/deeplabv3_seg_v4.ckpt'; \
json.dump(d, open(p, 'w')); print('[seg registry]', d)"

COPY core.py      /opt/app/
COPY inference.py /opt/app/
COPY src/         /opt/app/src/
COPY model_lib/   /opt/app/model_lib/
COPY pipeline/    /opt/app/pipeline/

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

ENTRYPOINT ["python", "inference.py"]
