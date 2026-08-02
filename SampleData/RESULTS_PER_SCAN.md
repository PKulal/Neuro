# Per-scan results on the 40 held-out patients

Every scan in this folder, scored with the shipped model.
**22 of 40 correct (55.0%).** Reproduce with `python test_model.py`.

## Scans the model gets RIGHT (22)

Use these if you need examples that demonstrate the application
working end to end.

They are **cherry-picked by outcome**. They are not a fair sample, and
quoting an accuracy figure from this subset alone would be
meaningless — it would say 100% for a model measured at 55%.

| File | Truth | Predicted | Autism prob | Confidence |
|---|---|---|---|---|
| `Autism_28860.nii.gz` | AUTISM | AUTISM | 81.1% | 96% |
| `Autism_28869.nii.gz` | AUTISM | AUTISM | 73.2% | 84% |
| `Autism_28874.nii.gz` | AUTISM | AUTISM | 70.1% | 96% |
| `Autism_28875.nii.gz` | AUTISM | AUTISM | 73.7% | 100% |
| `Autism_29006.nii.gz` | AUTISM | AUTISM | 58.2% | 60% |
| `Autism_29043.nii.gz` | AUTISM | AUTISM | 56.5% | 56% |
| `Autism_29104.nii.gz` | AUTISM | AUTISM | 61.2% | 84% |
| `Autism_29110.nii.gz` | AUTISM | AUTISM | 61.9% | 72% |
| `Autism_29112.nii.gz` | AUTISM | AUTISM | 57.8% | 56% |
| `Autism_30181.nii.gz` | AUTISM | AUTISM | 58.8% | 60% |
| `Healthy_28746.nii.gz` | HEALTHY | HEALTHY | 41.6% | 96% |
| `Healthy_28801.nii.gz` | HEALTHY | HEALTHY | 35.1% | 96% |
| `Healthy_28829.nii.gz` | HEALTHY | HEALTHY | 34.0% | 96% |
| `Healthy_28846.nii.gz` | HEALTHY | HEALTHY | 25.7% | 100% |
| `Healthy_28902.nii.gz` | HEALTHY | HEALTHY | 50.0% | 60% |
| `Healthy_29123.nii.gz` | HEALTHY | HEALTHY | 55.0% | 60% |
| `Healthy_29130.nii.gz` | HEALTHY | HEALTHY | 47.2% | 96% |
| `Healthy_29132.nii.gz` | HEALTHY | HEALTHY | 50.4% | 60% |
| `Healthy_29136.nii.gz` | HEALTHY | HEALTHY | 51.1% | 84% |
| `Healthy_30195.nii.gz` | HEALTHY | HEALTHY | 39.1% | 92% |
| `Healthy_30203.nii.gz` | HEALTHY | HEALTHY | 27.3% | 100% |
| `Healthy_30206.nii.gz` | HEALTHY | HEALTHY | 35.2% | 100% |

## Scans the model gets WRONG (18)

| File | Truth | Predicted | Autism prob | Confidence |
|---|---|---|---|---|
| `Autism_28752.nii.gz` | AUTISM | **HEALTHY** | 50.4% | 84% |
| `Autism_28778.nii.gz` | AUTISM | **HEALTHY** | 25.1% | 100% |
| `Autism_28789.nii.gz` | AUTISM | **HEALTHY** | 42.5% | 96% |
| `Autism_28821.nii.gz` | AUTISM | **HEALTHY** | 45.0% | 92% |
| `Autism_29009.nii.gz` | AUTISM | **HEALTHY** | 52.9% | 64% |
| `Autism_29053.nii.gz` | AUTISM | **HEALTHY** | 48.2% | 92% |
| `Autism_29115.nii.gz` | AUTISM | **HEALTHY** | 55.4% | 52% |
| `Autism_30176.nii.gz` | AUTISM | **HEALTHY** | 49.8% | 76% |
| `Autism_30185.nii.gz` | AUTISM | **HEALTHY** | 46.2% | 80% |
| `Autism_30188.nii.gz` | AUTISM | **HEALTHY** | 35.9% | 96% |
| `Healthy_28888.nii.gz` | HEALTHY | **AUTISM** | 67.6% | 84% |
| `Healthy_28892.nii.gz` | HEALTHY | **AUTISM** | 78.9% | 100% |
| `Healthy_28904.nii.gz` | HEALTHY | **AUTISM** | 60.3% | 64% |
| `Healthy_29020.nii.gz` | HEALTHY | **AUTISM** | 58.0% | 56% |
| `Healthy_29040.nii.gz` | HEALTHY | **AUTISM** | 62.2% | 64% |
| `Healthy_29054.nii.gz` | HEALTHY | **AUTISM** | 59.1% | 52% |
| `Healthy_30149.nii.gz` | HEALTHY | **AUTISM** | 57.4% | 56% |
| `Healthy_30193.nii.gz` | HEALTHY | **AUTISM** | 64.5% | 88% |

## Reading this table

The decision threshold is 0.562. Most scores sit between 0.45 and
0.65, so most verdicts are decided by a small margin — which is why
the AUC is 0.598, only a little above the 0.5 of random guessing.

The errors are not concentrated in unusual or corrupted scans. They
are spread evenly across sites and both classes, because the
underlying signal is weak rather than because particular files are
faulty.
