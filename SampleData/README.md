# Sample MRI scans

Four whole-brain MRI volumes so you can run the application immediately after
cloning, without downloading the full 16 GB ABIDE-II archive.

| File | True diagnosis | Site |
|---|---|---|
| `Autism_29006.nii.gz` | Autism | ABIDEII-BNI_1 |
| `Autism_29104.nii.gz` | Autism | ABIDEII-TCD_1 |
| `Healthy_29020.nii.gz` | Healthy | ABIDEII-BNI_1 |
| `Healthy_29130.nii.gz` | Healthy | ABIDEII-TCD_1 |

The true diagnosis is in each filename so you can check the model's answer.

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
