FROM python:3.7

# 2
RUN pip install Flask gunicorn fitter numpy opencv-python sympy plotly scipy matplotlib

# 3
COPY src/ /app
WORKDIR /app

# 4
ENV PORT 8080

# 5
CMD exec gunicorn --bind :$PORT --workers 2 --threads 8 app:app