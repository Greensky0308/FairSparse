"""Build the unified analysis cohort for TCGA-CHOL (rare liver tumor).

Merges patient clinical (patient_wide.tsv), sample-level features (sample_wide.tsv),
and molecular driver mutations (mutation_matrix.tsv) into cohort.tsv, and defines
the sparse molecular-subtype labels used as the fairness axis.
"""
import csv
import os

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def load_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    patients = {r["patientId"]: r for r in load_tsv(os.path.join(DATA, "patient_wide.tsv"))}
    samples = {r["sampleId"][:12]: r for r in load_tsv(os.path.join(DATA, "sample_wide.tsv"))}  # strip -01
    mutations = {r["patientId"]: r for r in load_tsv(os.path.join(DATA, "mutation_matrix.tsv"))}

    genes = ["IDH1", "IDH2", "FGFR2", "TP53", "KRAS", "BAP1", "ARID1A", "PBRM1",
             "SMAD4", "BRAF", "NRAS", "STK11", "CDKN2A", "PTEN", "PIK3CA"]

    # composite sparse molecular subtypes (canonical iCCA subtypes)
    def flag(pid, *gs):
        return int(any(mutations[pid][g] == "1" for g in gs if g in mutations[pid]))

    rows = []
    for pid, p in patients.items():
        s = samples.get(pid, {})
        m = mutations.get(pid, {})
        r = {
            "patientId": pid,
            # demographics
            "SEX": p.get("SEX", ""),
            "AGE": p.get("AGE", ""),
            "RACE": p.get("RACE", ""),
            "ETHNICITY": p.get("ETHNICITY", ""),
            # staging / grade
            "AJCC_STAGE": p.get("AJCC_PATHOLOGIC_TUMOR_STAGE", ""),
            "PATH_T": p.get("PATH_T_STAGE", ""),
            "PATH_N": p.get("PATH_N_STAGE", ""),
            "PATH_M": p.get("PATH_M_STAGE", ""),
            "GRADE": s.get("GRADE", p.get("GRADE", "")),
            # survival outcomes
            "OS_MONTHS": p.get("OS_MONTHS", ""),
            "OS_STATUS": p.get("OS_STATUS", ""),
            "DSS_MONTHS": p.get("DSS_MONTHS", ""),
            "DSS_STATUS": p.get("DSS_STATUS", ""),
            # genomic burden (sample-level)
            "MUTATION_COUNT": s.get("MUTATION_COUNT", ""),
            "TMB_NONSYNONYMOUS": s.get("TMB_NONSYNONYMOUS", ""),
            "FRACTION_GENOME_ALTERED": s.get("FRACTION_GENOME_ALTERED", ""),
            "ANEUPLOIDY_SCORE": s.get("ANEUPLOIDY_SCORE", ""),
            # anatomical subtype
            "TUMOR_TYPE": s.get("TUMOR_TYPE", ""),
        }
        # molecular subtype labels (fairness axis)
        r["IDH_mutant"] = flag(pid, "IDH1", "IDH2")
        r["FGFR2_mutant"] = flag(pid, "FGFR2")
        r["KRAS_mutant"] = flag(pid, "KRAS")
        r["BAP1_mutant"] = flag(pid, "BAP1")
        r["PBRM1_mutant"] = flag(pid, "PBRM1")
        r["TP53_mutant"] = flag(pid, "TP53")
        for g in genes:
            r[f"mut_{g}"] = int(m.get(g, "0") == "1")
        rows.append(r)

    cols = ["patientId", "SEX", "AGE", "RACE", "ETHNICITY", "AJCC_STAGE", "PATH_T",
            "PATH_N", "PATH_M", "GRADE", "OS_MONTHS", "OS_STATUS", "DSS_MONTHS",
            "DSS_STATUS", "MUTATION_COUNT", "TMB_NONSYNONYMOUS",
            "FRACTION_GENOME_ALTERED", "ANEUPLOIDY_SCORE", "TUMOR_TYPE",
            "IDH_mutant", "FGFR2_mutant", "KRAS_mutant", "BAP1_mutant",
            "PBRM1_mutant", "TP53_mutant"] + [f"mut_{g}" for g in genes]

    out = os.path.join(DATA, "cohort.tsv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"cohort.tsv: {len(rows)} patients x {len(cols)} columns")
    print("\n--- rare subtype sizes (fairness axis) ---")
    for label in ["IDH_mutant", "FGFR2_mutant", "KRAS_mutant", "BAP1_mutant",
                  "PBRM1_mutant", "TP53_mutant"]:
        n = sum(1 for r in rows if r[label] == 1)
        print(f"  {label:16s} n={n:2d} ({100*n/len(rows):.0f}%)")
    from collections import Counter
    print("\n--- anatomic subtypes ---")
    for k, v in Counter(r["TUMOR_TYPE"] for r in rows).most_common():
        print(f"  {k}: {v}")
    print("\n--- OS_STATUS ---")
    print(" ", Counter(r["OS_STATUS"] for r in rows))


if __name__ == "__main__":
    main()
