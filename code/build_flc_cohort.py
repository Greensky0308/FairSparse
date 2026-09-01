"""Build FLC cohort from PMC10983114 Table S1 (manually curated from PDF).

Features: AGE (numeric), SEX (F/M), Sample_Type (Primary/Metastatic/Recurrent),
Sample_Location, FUSION_NEOEPITOPES (numeric), + binary subtype flags.
Note: FLC has no survival outcome in this table -> used for subtype-fidelity
analysis only (excluded from survival-utility).
"""
import os
import csv

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "flc")

# patientId, SEX, AGE, Sample_Type, Sample_Location, FUSION_NEOEPITOPES
PATIENTS = [
    ("FLC01", "F", 24, "Metastatic", "Unknown", 2),
    ("FLC02", "M", 32, "Metastatic", "Extrahepatic", 3),
    ("FLC03", "M", 19, "Metastatic", "Extrahepatic", 0),
    ("FLC04", "M", 27, "Metastatic", "Extrahepatic", 3),
    ("FLC05", "F", 25, "Metastatic", "Lymph node", 7),
    ("FLC06", "F", 18, "Recurrent", "Liver", 10),
    ("FLC07", "M", 16, "Primary", "Liver", 3),
    ("FLC09", "M", 28, "Primary", "Liver", 5),
    ("FLC12", "M", 27, "Recurrent", "Liver", 9),
    ("FLC13", "F", 17, "Metastatic", "Lung", 4),
    ("FLC15", "F", 48, "Metastatic", "Ascites", 0),
    ("FLC17", "M", 17, "Metastatic", "Lymph node", 10),
    ("FLC18", "F", 15, "Primary", "Liver", 5),
    ("FLC20", "M", 19, "Metastatic", "Lymph node", 3),
    ("FLC23", "F", 49, "Primary", "Liver", 3),
    ("FLC25", "F", 22, "Metastatic", "Lymph node", 7),
    ("FLC26", "M", 18, "Primary", "Liver", 2),
    ("FLC27", "F", 18, "Metastatic", "Lymph node", 7),
    ("FLC29", "F", 29, "Metastatic", "Lung", 3),
    ("FLC30", "F", 31, "Metastatic", "Peritoneal", 3),
]

cols = ["patientId", "SEX", "AGE", "Sample_Type", "Sample_Location",
        "FUSION_NEOEPITOPES", "RECURRENT", "PRIMARY"]
os.makedirs(DATA, exist_ok=True)
with open(os.path.join(DATA, "cohort.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(cols)
    for pid, sex, age, stype, sloc, neop in PATIENTS:
        w.writerow([pid, sex, age, stype, sloc, neop,
                    1 if stype == "Recurrent" else 0,
                    1 if stype == "Primary" else 0])

print(f"FLC cohort: {len(PATIENTS)} patients")
from collections import Counter
print("Sample_Type:", Counter(p[3] for p in PATIENTS))
print("SEX:", Counter(p[1] for p in PATIENTS))
