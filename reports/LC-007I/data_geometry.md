# Cached data and native crop exposure

CPU decoded all 400 cached B frames. Train360 contains 44,856 GT and internal-val40 contains 4,295 GT. All decoded frames are 1024×768 and agree with the manifest. GT rows are finite and within the original frame. The manifest SHA-256 is `de60892266c08bce96a3ea07e884c86eec8d05e0a9d31c0949bcff9e925fe482`. The cached ResNet-18 SHA-256 is `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. [Cache and initialization audit](initialization_audit.json)

Both crop sizes are evaluated on the same 360 training images and minibatch order in each of 40 epochs. The 512 trajectory reproduces the archived crop and LR hashes and sampled GT, overflow, positive slots and negative slots. The 768 trajectory retains the image order and LR hash while its crop hash includes the requested size. Origin changes follow the same seed and original random crop rule. Crop GT remains at native coordinates. Cell overflow counts annotations beyond the two slots in stride-4 cells.

| Training crop | Valid pixels over 40 epochs | Padding pixels | Sampled GT | GT exceeding slots |
| --- | ---: | ---: | ---: | ---: |
| 512×512 | 3,774,873,600 | 0 | 415,385 | 60 |
| 768×768 | 8,493,465,600 | 0 | 1,395,609 | 180 |

The paired exposure CSV lists per-epoch totals and positive/negative slot counts. Sampled GT includes repeated exposure across epochs. Synthetic regression covers a smaller irregular frame, neutral right/bottom padding, stride-4 validity masking and colliding points. The 512 validation tile and its existing ownership boundaries remain identical. [Paired exposure](crop_exposure.csv)
