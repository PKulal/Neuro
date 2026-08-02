# Test MRI scans

All **40 held-out test patients** — 20 Autism and 20 Healthy — so the
application can be run and evaluated immediately after cloning, without
downloading the full 16 GB ABIDE-II archive.

```
Autism_28752   Autism_28778   Autism_28789   Autism_28821   Autism_28860
Autism_28869   Autism_28874   Autism_28875   Autism_29006   Autism_29009
Autism_29043   Autism_29053   Autism_29104   Autism_29110   Autism_29112
Autism_29115   Autism_30176   Autism_30181   Autism_30185   Autism_30188

Healthy_28746  Healthy_28801  Healthy_28829  Healthy_28846  Healthy_28888
Healthy_28892  Healthy_28902  Healthy_28904  Healthy_29020  Healthy_29040
Healthy_29054  Healthy_29123  Healthy_29130  Healthy_29132  Healthy_29136
Healthy_30149  Healthy_30193  Healthy_30195  Healthy_30203  Healthy_30206
```

Drawn 4 Autism + 4 Healthy from each of the five sites, so no single scanner
dominates. The true diagnosis is in each filename so you can check the model's
answer.

## What "working correctly" means here

All 40 files load, validate and produce a patient-level result — verified,
40/40, no errors.

**The model's answers are a different matter: 22 of 40 are correct (55%).** The
software is functioning exactly as designed; the model simply has not learned a
strong signal. Expect roughly 4 or 5 wrong answers out of every 10 scans. See
the accuracy discussion in the main README before drawing conclusions from any
single prediction.

## How to use them

```powershell
# GUI: click "Select Brain MRI", navigate here, pick a file, click PREDICT
.\.venv\Scripts\python.exe app.py

# Command line
python test_model.py --mri SampleData\Autism_29006.nii.gz
```

## These four patients were never trained on

All four come from the 40 held-out test patients — excluded from training and
from every tuning decision. A prediction on these is a genuine test of
generalisation.

## Do not judge the model from four scans

Measured held-out accuracy is **45%** across all 40 patients, with an AUC of
0.565 — close to chance. Individual answers on these samples will frequently be
wrong, and a correct answer on one scan is luck rather than evidence. For the
real evaluation run `python test_model.py`, which scores all 40.

## Provenance and licence

These files are unmodified anatomical scans from **ABIDE-II** (Autism Brain
Imaging Data Exchange II), redistributed under the licence ABIDE-II is released
under: **Creative Commons Attribution-NonCommercial-ShareAlike**.

Terms that carry over to anyone using them:

- **Attribution** — cite ABIDE-II and the contributing sites
- **NonCommercial** — research and educational use only
- **ShareAlike** — redistribute only under the same licence

The data is anonymised in accordance with HIPAA guidelines and INDI protocols;
it contains no protected health information.

Full dataset and citation requirements:
<http://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html>

If your use is commercial, remove this folder.

## Getting the other 36

If you have registered for ABIDE-II and downloaded it, collect every held-out
patient's whole MRI into one folder:

```powershell
python export_test_mri.py
```
