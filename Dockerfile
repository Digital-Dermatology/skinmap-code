FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# System deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      apt-transport-https libtcmalloc-minimal4 libomp-dev sox git gcc g++ python3-dev \
      ffmpeg libsm6 libxext6 texlive-latex-base texlive-latex-extra texlive-fonts-recommended \
      xzdec dvipng cm-super libglib2.0-0 libgl1-mesa-glx \
 && rm -rf /var/lib/apt/lists/*

# Helpful pip env
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1

# Timezone (optional)
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime && echo "Etc/UTC" > /etc/timezone

# Keep pip itself current
RUN python -m pip install --upgrade pip wheel setuptools

WORKDIR /workspace
RUN mkdir -p /assets

# 1) Provide constraints that pin NumPy below 2
COPY constraints.txt /assets/constraints.txt

# 2) Preinstall the compiled bits in the right order under constraints:
#    - Pin NumPy < 2 to match FAISS wheels that expect NumPy 1.x
#    - Install faiss-gpu via conda (better support for Python 3.11+)
RUN python -m pip install "numpy<2" -c /assets/constraints.txt
RUN conda install -y -c pytorch -c conda-forge libstdcxx-ng faiss-gpu && conda clean -afy
ENV LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH}

# 3) Install the rest of requirements, still honoring constraints
COPY requirements.txt /assets/requirements.txt
RUN python -m pip install -r /assets/requirements.txt -c /assets/constraints.txt

# Your code
COPY . /workspace/
RUN git config --global --add safe.directory '*'

ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONPATH=/workspace

# Smoke test at build time so the image fails early if anything drifts
RUN python - <<'PY'
import numpy, faiss
print("NumPy:", numpy.__version__)
print("FAISS:", faiss.__version__)
PY

EXPOSE 8888
