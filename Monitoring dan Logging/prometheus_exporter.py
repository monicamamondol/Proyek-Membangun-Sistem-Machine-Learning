from flask import Flask, request, jsonify, Response
import requests
import time
import psutil
import gc
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST
)

app = Flask(__name__)

# Buat registry khusus agar tidak terjadi duplikasi
registry = CollectorRegistry()

# --- 1. Total HTTP Requests ---
http_requests_total = Counter("http_requests_total", "Total HTTP Requests to ML Model", registry=registry)

# --- 2. Total Request Throughput ---
http_requests_throughput_total = Counter("http_requests_throughput_total", "Total Request Throughput", registry=registry)

# --- 3 & 4. Histogram untuk Latency (otomatis menghasilkan count & sum) ---
http_request_duration_seconds = Histogram("http_request_duration_seconds", "HTTP Request Latency (seconds)", registry=registry)

# --- 6. CPU Usage ---
system_cpu_usage = Gauge("system_cpu_usage", "System CPU Usage (%)", registry=registry)

# --- 7. RAM Usage ---
system_ram_usage = Gauge("system_ram_usage", "System RAM Usage (%)", registry=registry)

# --- 8. GC Objects Collected per Generation ---
python_gc_objects_collected_total = Gauge(
    "python_gc_objects_collected_total", "Total GC Objects Collected by Python", ['generation'], registry=registry)

# --- 9. GC Collections Total per Generation ---
python_gc_collections_total = Counter(
    "python_gc_collections_total", "Python GC Collections Count", ['generation'], registry=registry)

# --- 10. Python Info ---
python_info = Info("python_info", "Python Runtime Info", registry=registry)
python_info.info({
    "implementation": "CPython",
    "version": "3.13.3"
})


@app.route('/metrics', methods=['GET'])
def metrics():
    # Update CPU & RAM
    system_cpu_usage.set(psutil.cpu_percent(interval=1))
    system_ram_usage.set(psutil.virtual_memory().percent)

    # Update GC info
    counts = gc.get_count()
    for gen in range(3):
        python_gc_objects_collected_total.labels(generation=str(gen)).set(counts[gen])
        python_gc_collections_total.labels(generation=str(gen)).inc()

    return Response(generate_latest(registry), mimetype=CONTENT_TYPE_LATEST)


@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    http_requests_total.inc()
    http_requests_throughput_total.inc()

    data = request.get_json()
    api_url = "http://127.0.0.1:5005/invocations"

    try:
        response = requests.post(api_url, json=data)
        duration = time.time() - start_time
        http_request_duration_seconds.observe(duration)

        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)
