"""
Comprehensive test script for dataset upload and validation.
Tests:
  1. Valid CSV with 'label' (0 and 1)
  2. Missing 'label' column with NO candidates
  3. Missing 'label' column WITH candidate ('class' / 'attack')
  4. Mapping candidate target column to 'label'
  5. Invalid label values (e.g. 2, 99)
  6. Download sample template
"""
import requests
import io
import pandas as pd

BASE_URL = "http://localhost:8000"

# 1. Login
print("--- 1. Authenticating ---")
login_resp = requests.post(
    f"{BASE_URL}/api/auth/login",
    json={"email": "demo@gnss.edu", "password": "demo1234"}
)
assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
token = login_resp.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
print("[PASS] Logged in as demo@gnss.edu")

# 2. Download Sample Template
print("\n--- 2. Testing Template Download ---")
template_resp = requests.get(f"{BASE_URL}/api/dataset/sample-template", headers=headers)
assert template_resp.status_code == 200, f"Template download failed: {template_resp.status_code}"
assert "label" in template_resp.text, "Template missing 'label' column"
print(f"[PASS] Template downloaded successfully ({len(template_resp.text)} bytes)")

# 3. Test Valid Dataset Upload
print("\n--- 3. Testing Valid Dataset Upload ---")
valid_csv = """timestamp,pseudorange,cn0,latitude,longitude,altitude,label
1000.0,20000100.0,42.0,28.61,77.20,216.0,0
1001.0,20000105.0,42.1,28.61,77.20,216.1,0
1002.0,20000600.0,28.0,28.65,77.25,230.0,1
1003.0,20000650.0,27.5,28.66,77.26,234.0,1
"""
files = {"file": ("test_valid.csv", io.BytesIO(valid_csv.encode("utf-8")), "text/csv")}
resp = requests.post(f"{BASE_URL}/api/dataset/upload", headers=headers, files=files)
assert resp.status_code == 200, f"Valid upload failed: {resp.text}"
data = resp.json()
print(f"[PASS] Valid dataset uploaded: ID={data['dataset_id']}, Shape={data['shape']}, Labels={data['label_distribution']}")

# 4. Test Missing 'label' Column (No candidates)
print("\n--- 4. Testing Missing Label Column Error ---")
no_label_csv = """timestamp,pseudorange,cn0,latitude,longitude,altitude
1000.0,20000100.0,42.0,28.61,77.20,216.0
1001.0,20000105.0,42.1,28.61,77.20,216.1
"""
files = {"file": ("test_no_label.csv", io.BytesIO(no_label_csv.encode("utf-8")), "text/csv")}
resp = requests.post(f"{BASE_URL}/api/dataset/upload", headers=headers, files=files)
assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
err = resp.json()["detail"]
print(f"[PASS] Missing label rejected correctly:")
print(f"       Error: {err.get('message', err)}")
print(f"       Shape returned: {err.get('shape')}")
print(f"       Columns returned: {err.get('columns')}")

# 5. Test Alternative Target Column Detection (e.g. 'attack')
print("\n--- 5. Testing Alternative Target Candidate Detection ('attack') ---")
candidate_csv = """timestamp,pseudorange,cn0,latitude,longitude,altitude,attack
1000.0,20000100.0,42.0,28.61,77.20,216.0,0
1001.0,20000105.0,42.1,28.61,77.20,216.1,0
1002.0,20000600.0,28.0,28.65,77.25,230.0,1
"""
files = {"file": ("test_candidate.csv", io.BytesIO(candidate_csv.encode("utf-8")), "text/csv")}
resp = requests.post(f"{BASE_URL}/api/dataset/upload", headers=headers, files=files)
assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
err = resp.json()["detail"]
assert "attack" in err.get("detected_candidates", []), f"Failed to detect candidate: {err}"
print(f"[PASS] Candidate detected correctly: {err.get('detected_candidates')}")

# 6. Test Mapping Target Column ('attack' -> 'label')
print("\n--- 6. Testing Explicit Target Column Mapping ---")
files = {"file": ("test_candidate.csv", io.BytesIO(candidate_csv.encode("utf-8")), "text/csv")}
data_form = {"target_column": "attack"}
resp = requests.post(f"{BASE_URL}/api/dataset/upload", headers=headers, files=files, data=data_form)
assert resp.status_code == 200, f"Mapping upload failed: {resp.text}"
data = resp.json()
assert data["mapped_target_column"] == "attack", f"Expected mapped column attack: {data}"
print(f"[PASS] Column 'attack' mapped to 'label' successfully! ID={data['dataset_id']}, Shape={data['shape']}")

# 7. Test Invalid Label Values (e.g. 2, 99)
print("\n--- 7. Testing Invalid Label Values Rejection ---")
invalid_labels_csv = """timestamp,pseudorange,cn0,label
1000.0,20000100.0,42.0,0
1001.0,20000105.0,42.1,1
1002.0,20000600.0,28.0,2
1003.0,20000650.0,27.5,99
"""
files = {"file": ("test_invalid_vals.csv", io.BytesIO(invalid_labels_csv.encode("utf-8")), "text/csv")}
resp = requests.post(f"{BASE_URL}/api/dataset/upload", headers=headers, files=files)
assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
err = resp.json()["detail"]
assert "invalid_values" in err or "Invalid" in str(err), f"Expected invalid values detail: {err}"
print(f"[PASS] Invalid labels rejected correctly: {err.get('message', err)}")

# 8. Test Prediction without Label Column
print("\n--- 8. Testing Prediction without Label Column ---")
unlabelled_gnss_csv = """timestamp,pseudorange,carrier_phase,cn0,doppler,elevation,azimuth,latitude,longitude,altitude,velocity
1710000000.0,21450123.4,112717411.2,42.5,-2.1,65.4,134.2,28.61390,77.20900,216.0,0.12
1710000001.0,21450125.1,112717420.1,42.1,-2.3,65.4,134.3,28.61391,77.20901,216.1,0.15
1710000002.0,21450127.8,112717434.3,42.4,-2.0,65.5,134.4,28.61390,77.20900,215.9,0.11
1710000003.0,21450129.5,112717443.0,43.0,-1.9,65.5,134.5,28.61392,77.20902,216.2,0.18
1710000004.0,21450131.2,112717451.9,42.8,-2.2,65.6,134.6,28.61391,77.20901,216.0,0.14
1710000005.0,21450133.0,112717461.4,42.2,-2.1,65.6,134.7,28.61390,77.20900,215.8,0.13
1710000006.0,21450134.7,112717470.2,42.6,-2.0,65.7,134.8,28.61391,77.20901,216.1,0.16
1710000007.0,21450136.5,112717479.8,42.9,-2.4,65.7,134.9,28.61390,77.20900,215.9,0.10
1710000008.0,21450138.3,112717488.5,42.3,-2.2,65.8,135.0,28.61392,77.20902,216.3,0.17
1710000009.0,21450140.1,112717497.1,42.7,-2.1,65.8,135.1,28.61391,77.20901,216.0,0.12
1710000010.0,21450142.0,112717506.0,42.4,-2.2,65.9,135.2,28.61390,77.20900,216.0,0.14
1710000011.0,21450143.8,112717514.8,42.8,-2.0,65.9,135.3,28.61391,77.20901,216.1,0.15
1710000012.0,21450145.5,112717523.5,42.2,-2.3,66.0,135.4,28.61390,77.20900,215.8,0.11
1710000013.0,21450147.2,112717532.1,42.6,-2.1,66.0,135.5,28.61392,77.20902,216.2,0.16
1710000014.0,21450149.0,112717541.0,42.9,-2.4,66.1,135.6,28.61391,77.20901,216.0,0.13
1710000015.0,21450150.8,112717549.9,42.3,-2.2,66.1,135.7,28.61390,77.20900,215.9,0.12
1710000016.0,21450152.6,112717558.8,42.7,-2.0,66.2,135.8,28.61391,77.20901,216.1,0.17
1710000017.0,21450154.4,112717567.5,42.5,-2.1,66.2,135.9,28.61390,77.20900,215.8,0.11
1710000018.0,21450156.2,112717576.2,42.1,-2.3,66.3,136.0,28.61392,77.20902,216.3,0.15
1710000019.0,21450158.0,112717585.0,42.8,-2.0,66.3,136.1,28.61391,77.20901,216.0,0.14
1710000020.0,21450159.9,112717594.0,42.4,-2.2,66.4,136.2,28.61390,77.20900,215.9,0.12
"""
files = {"file": ("unlabelled_flight.csv", io.BytesIO(unlabelled_gnss_csv.encode("utf-8")), "text/csv")}
resp = requests.post(f"{BASE_URL}/api/predict", headers=headers, files=files)
assert resp.status_code == 200, f"Unlabelled predict failed: {resp.text}"
pred_data = resp.json()
print(f"[PASS] Unlabelled prediction succeeded: Result={pred_data['overall_prediction']}, Confidence={pred_data['confidence']}%")

print("\n==============================================")
print("ALL 8 DATASET VALIDATION TESTS PASSED 100%!")
print("==============================================")
