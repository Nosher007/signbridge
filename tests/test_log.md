# SignBridge — Test Log

> Every task gets a test. Log results here before marking a task done.

---

## Day 1 — Setup + Data + EDA

- [x] GCP setup → `gsutil ls gs://signbridge-data/` ✅ — project `signbridge-prod` created, billing linked, 6 APIs enabled, bucket confirmed
- [x] GCP VM setup ✅ — `signbridge-vm` (n2-standard-4, us-central1-a) running, SSH confirmed, GCS accessible from VM. Training on Kaggle free T4 (GPU exhausted globally); CPU VM used for data/EDA work.
- [ ] ASL dataset upload → file count confirmed in GCS
- [ ] WLASL dataset upload → file count confirmed in GCS
- [ ] ASL EDA notebook → all cells run, class dist plot saved
- [ ] WLASL EDA notebook → all cells run, frame dist plot saved
- [ ] `pip install -r requirements.txt` → no errors
- [ ] GitHub repo initial commit → correct folder structure visible
