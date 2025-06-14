import requests
import json
import time
import logging
 
# Konfigurasi logging untuk menulis ke file txt
logging.basicConfig(
    filename='api_monitoring_logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
 
# Endpoint API model
# Setelah diubah
API_URL = "http://127.0.0.1:8000/predict"

 
# Kolom dan data model
columns = [
    "id", "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean",
    "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se",
    "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
    "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst", "Unnamed: 32"
]

data = [[
    -0.23712699397470832, -1.4407529621170216, -0.43531946993557874,
    -1.362084967747501, -1.1391178966662039, 0.7805733144944511,
    0.7189212819653799, 2.8231345141359507, -0.11914956451814925,
    1.0926621933891159, 2.458172609472401, -0.26380039208863376,
    -0.01605245548331523, -0.47041357205951123, -0.4747608837338361,
    0.8383649314058185, 3.251026914353816, 8.438936670491492,
    3.3919873274645718, 2.6211657407324096, 2.0612078679401953,
    -1.2328613112975744, -0.47630949192646244, -1.2479200916849496,
    -0.9739675822549873, 0.7228944471170092, 1.1867323172434845,
    4.67282795901713, 0.9320124023116306, 2.0972421679802107,
    1.886450144695108, 0
]]

# Buat payload sesuai format yang dibutuhkan model
input_data = {
    "dataframe_split": {
        "columns": columns,
        "data": data
    }
}

# Konversi ke JSON
headers = {"Content-Type": "application/json"}
payload = json.dumps(input_data)
 
# Mulai mencatat waktu eksekusi
start_time = time.time()
 
for i in range(30):
    try:
        start_time = time.time()
        
        # Kirim request ke API
        response = requests.post(API_URL, headers=headers, data=payload)
        
        # Hitung response time
        response_time = time.time() - start_time

        if response.status_code == 200:
            prediction = response.json()
            logging.info(f"[{i+1}] Response: {prediction}, Time: {response_time:.4f} sec")
            print(f"[{i+1}] Prediction: {prediction} | Time: {response_time:.4f} sec")
        else:
            logging.error(f"[{i+1}] Error {response.status_code}: {response.text}")
            print(f"[{i+1}] Error {response.status_code}: {response.text}")

        time.sleep(0.2)  # opsional, biar tidak terlalu cepat

    except Exception as e:
        logging.error(f"[{i+1}] Exception: {str(e)}")
        print(f"[{i+1}] Exception: {str(e)}")
