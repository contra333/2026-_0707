# Issue 6 OpenOOD CIFAR-10 Server Validation

## Validation context

- Validation date: 2026-07-13
- Hostname: `curie`
- Conda environment: `oge-issue6`
- Repository branch: `feat/issue-6-openood-cifar10-protocol`
- Repository commit: `68e6eaf408124468384c5f5df118a5dc8426462e`
- OpenOOD source commit: `3c35632ee91b54b09d1f085d04f94744cece7d0b`
- Final data root: `/home/ghjin/datasets/openood-v1.5-3c35632e`
- Server artifact root: `/home/ghjin/0707_exp/issue6_artifacts/68e6eaf`

The validation used the released OpenOOD archives downloaded from the pinned
Google Drive file IDs. Existing datasets elsewhere under
`/home/ghjin/datasets` were not reused or modified.

## Runtime environment

| component | observed value |
| --- | --- |
| Python | `3.13.11` |
| PyTorch | `2.11.0+cu130` |
| TorchVision | `0.26.0+cu130` |
| CUDA reported by PyTorch | `13.0` |
| cuDNN reported by PyTorch | `91900` (cuDNN 9.19.0) |
| GPU | 4 x NVIDIA RTX A5000, 24,564 MiB each |
| NVIDIA driver | `580.119.02` |

The environment evidence is stored outside Git at
`/home/ghjin/0707_exp/issue6_artifacts/68e6eaf/environment/`. It contains
`conda-explicit.txt`, `conda-environment-no-builds.yaml`, `pip-freeze.txt`,
`nvidia-smi.txt`, `git-sha.txt`, and `runtime.txt`.

## Test validation

The editable package was installed in `oge-issue6` without dependency changes,
and the complete repository test suite was run with:

```bash
conda run -n oge-issue6 pytest -q
```

Result: `53 passed in 3.56s`.

## Released archive validation

Each archive was downloaded sequentially, retained under
`/home/ghjin/datasets/openood-v1.5-3c35632e-archives`, and checked with the
`file` command, byte size, SHA256, `zipinfo -t`, entry enumeration, unsafe-path
checks, and expected uncompressed size. All eight archives passed ZIP integrity
validation and a later `sha256sum -c SHA256SUMS` revalidation.

| logical name | Google Drive file ID | bytes | local SHA256 | ZIP status |
| --- | --- | ---: | --- | --- |
| `benchmark_imglist` | `1lI1j0_fDDvjIt9JlWAw09X8ks-yrR_H1` | 27,962,573 | `647b8646327906c33e9d41294789984b9625a021968aec76e34fd521109d138e` | validated |
| `cifar10` | `1Co32RiiWe16lTaiOU6JMMnyUYS41IlO1` | 142,903,414 | `5d3c480cd13e8791af7429fde3884f6ac45a7191c17fc1414eec550ed2e6582c` | validated |
| `cifar100` | `1PGKheHUsf29leJPPGuXqzLBMwl8qMF8_` | 141,321,419 | `db6301142ca4119cb104a194e31a8f1190a1873759c3aa409b04c7899e9868f8` | validated |
| `tin` | `1PZ-ixyx52U989IKsMA2OT-24fToTrelC` | 237,497,520 | `e95af0741e02afeb62c58cac3b5ac53ada99bee67bde65b061a423e2e40deb9d` | validated |
| `mnist` | `1CCHAGWqA1KJTFFswuF9cbhmB-j98Y1Sb` | 47,228,895 | `47e04388bccdcb0c5bb416e98720382129156c13f56dadeb97935f4a012cd0f9` | validated |
| `svhn` | `1DQfc11HOtB1nEwqS4pWUFp8vtQ3DczvI` | 18,989,682 | `56fdae7a5409712bcf10ce460c3e1f30550e73c89b70b9544bf136a84c61c0ae` | validated |
| `texture` | `1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam` | 625,703,847 | `801fd2026c281090e6f42a5bddbc7ab55d8d03e79ee4bcc22d61bdef59862926` | validated |
| `places365` | `1Ec-LRSTf6u5vEctKX9vRp9OA6tqnJ0Ay` | 496,937,997 | `76639b253baca242da1b2746b56983c364e999d0eb63c4b901d8473f23b2e7ab` | validated |

Archives were extracted into separate staging directories before the final root
was assembled without image re-encoding, resizing, or filename changes.
Representative images from every image archive decoded successfully. The final
root contains 378,573 regular files, including 378,172 image files.

## Imglist and actual-data manifest validation

The repository validator was run as:

```bash
python scripts/verify_openood_data.py \
  --data-root /home/ghjin/datasets/openood-v1.5-3c35632e \
  --dataset cifar10 \
  --write-report /home/ghjin/0707_exp/issue6_artifacts/68e6eaf/openood_cifar10_manifest.json
```

| role | dataset | count | imglist SHA256 | label range |
| --- | --- | ---: | --- | --- |
| ID train | CIFAR-10 | 50,000 | `9317e18980f1b11df74bba9d64223e57e69bd08c2943adbc75aec393a4976d1b` | 0 to 9 |
| ID validation | CIFAR-10 | 1,000 | `6f64c4370bf3c61dab3308f7e8c5c94938b112b20d38cca77c1dfa0bcc5c1d89` | 0 to 9 |
| ID test | CIFAR-10 | 9,000 | `84559ec5225425e7f7363058d5f9442ed21a39c7c66b63c3f1bb8b1ddb935e13` | 0 to 9 |
| compatibility-only OOD validation | TinyImageNet | 1,000 | `39e22c5c6c2d95f95ab5ba43106f43ad7aa254ff62316104f406f7f10811f7c2` | -1 |
| near-OOD | CIFAR-100 | 9,000 | `f907cb13024ba0fd13099954064df8f2f1ab8e3369c27b95ab44f3c5fc87ad16` | -1 |
| near-OOD | TinyImageNet | 7,793 | `a44fd34e3c925d984063d2ad64969ff71e8fa6f5524fbe9da481c8b0c87b27cd` | -1 |
| far-OOD | MNIST | 70,000 | `48f8055d7327cbe91bff3ab6b43f9b623c3983355c55c246207907841c2ccb92` | -1 |
| far-OOD | SVHN | 26,032 | `f0cb7b825a985af2a6bf803cf51d7ca004e24ee8d8051eb6805bf77c77621c1e` | -1 |
| far-OOD | Textures | 5,640 | `644fcaec364ee1ee440ff5285e1581d162b957378bb240ac4b7fc26e4f2f48e4` | -1 |
| far-OOD | Places365 | 35,195 | `fa127faa5c1bbc5b567e1cd2871b04b570f045d1676550f441ec5e7c91ab6cc7` | -1 |

All ten manifests had zero missing images and zero duplicate sample IDs. The ID
label ranges were within `[0, 9]`; every released OOD list used label `-1`. The
validator reported an empty `errors` list.

## CUDA MSP vertical slice

The actual-data CUDA smoke command was:

```bash
python scripts/eval_ood_smoke.py \
  --data-root /home/ghjin/datasets/openood-v1.5-3c35632e \
  --model wrn28_10 \
  --device cuda \
  --id-max-samples 128 \
  --ood-max-samples 128 \
  --output-dir /home/ghjin/0707_exp/issue6_artifacts/68e6eaf/openood_msp_smoke
```

The loader-to-WRN-28-10-to-MSP-to-metrics-to-artifacts vertical slice completed
successfully on CUDA. It produced 128 unique score records for ID test and for
each near/far OOD dataset. JSON, YAML, and NPZ schema checks passed. Run metadata
records `smoke_only=true` and `model_is_random_or_untrained=true`; these random
model metrics are infrastructure validation only and are not research evidence.

## Server artifacts retained outside Git

- `/home/ghjin/datasets/openood-v1.5-3c35632e-archives/SHA256SUMS`
- `/home/ghjin/0707_exp/issue6_artifacts/68e6eaf/download_manifest.json`
- `/home/ghjin/0707_exp/issue6_artifacts/68e6eaf/extraction_manifest.json`
- `/home/ghjin/0707_exp/issue6_artifacts/68e6eaf/openood_cifar10_manifest.json`
- `/home/ghjin/0707_exp/issue6_artifacts/68e6eaf/artifact_schema_report.json`
- `/home/ghjin/0707_exp/issue6_artifacts/68e6eaf/openood_msp_smoke/`
- `/home/ghjin/0707_exp/issue6_artifacts/68e6eaf/environment/`

Large ZIP archives, extracted datasets, NPZ score files, and complete logs are
not committed to Git.

## Remaining limitations

- The pinned Google Drive release does not provide an independently published
  upstream checksum against which the locally preserved SHA256 values can be
  authenticated.
- ZIP integrity covered every entry, but full image decode was not run; decode
  checks covered representative images from each image archive.
- The official released TinyImageNet and Places365 list membership was used
  unchanged, but the semantic-overlap-removal generation process was not
  reconstructed.
- The random-model smoke did not fix an initialization seed. Its metric values
  are not reproducible research results.
