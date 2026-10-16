FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# torch รุ่น CPU-only: API นี้รันบน CPU (device = cpu ใน app.py)
# ถ้า pip install torch ตรง ๆ จาก PyPI บน Linux จะได้รุ่น CUDA พ่วง nvidia-* หลายตัว ทำให้ image ใหญ่มากโดยไม่จำเป็น
# แยกเป็น layer ของตัวเอง เพื่อให้ cache ไว้ได้ ไม่ต้องโหลดใหม่ทุกครั้งที่แก้โค้ด
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3"

COPY requirements.txt .
RUN pip install -r requirements.txt

# รันด้วย user ธรรมดา ไม่ใช้ root
RUN useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser . .
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/' % os.getenv('PORT','8000'), timeout=4)"

# ใช้ $PORT ถ้า platform กำหนดมา (Cloud Run / Render / Railway ฯลฯ) ไม่งั้นใช้ 8000 เหมือนเดิม
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
