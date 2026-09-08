# Image du service ANPR.
#
# AVERTISSEMENT : ce Dockerfile n'a pas ete construit ni execute. Docker n'etait
# pas installe sur la machine de developpement. Il est fourni parce qu'il fait
# partie du cahier des charges et qu'il documente les dependances systeme
# reelles, mais il doit etre considere comme non valide tant qu'un
# `docker build` n'a pas abouti.

FROM python:3.10-slim

# OpenCV compile sans interface graphique a tout de meme besoin de ces
# bibliotheques systeme pour decoder images et videos.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Les dependances sont installees avant le code : une modification du code
# n'invalide alors pas le cache de cette couche, qui est la plus longue.
COPY requirements.txt ./
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/
COPY pyproject.toml ./
RUN pip install --no-cache-dir --no-deps -e .

# Les poids ne sont pas copies dans l'image : ils sont montes en volume.
# Les embarquer alourdirait l'image de plusieurs centaines de megaoctets et
# imposerait une reconstruction a chaque reentrainement.
ENV PYTHONUNBUFFERED=1 \
    ANPR_PLATE_WEIGHTS=/app/data/models/plate_detector.pt

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "anpr.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
