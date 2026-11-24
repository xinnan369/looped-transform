# NWPU-Crowd first-shard provenance

This review inspected the existing archive's Windows metadata, ZIP central directory and SHA-256. Its JPEG entries stayed compressed. LC-002R and this review made no NWPU image transfers.

| Item | Local evidence |
|---|---|
| Existing archive | `D:\datasets\images_part1.zip` |
| Archive length | 3,722,384,133 bytes |
| NTFS allocation | 3,722,387,456 bytes, reported by Windows `FileStandardInfo.AllocationSize` |
| Expanded size recorded in the ZIP | 3,846,413,481 logical bytes across 1,000 JPEG entries. Actual extracted allocation was not measured. Keeping both would require at least 7,568,797,614 logical bytes, excluding filesystem overhead. |
| Archive SHA-256 | `f27e4371323f0251ef3828df00a6603abbfaf1ccb42be285fa94e5bf3ec05345` |
| NTFS creation time | 2025-12-03 19:44:57.100 China Standard Time (2025-12-03 11:44:57.100 UTC) |
| NTFS last write | 2025-12-03 19:51:36 China Standard Time (2025-12-03 11:51:36 UTC) |
| Release reference | [Author's images_part1.zip](https://huggingface.co/datasets/Junyu3/NWPU-Crowd/blob/main/images_part1.zip). The previously recorded published LFS digest and byte count match this local file. |
| Original acquisition source and time | Unverified. The archive has no `Zone.Identifier` stream and the available project records have no original transfer log. The NTFS timestamps describe this local file. They do not identify the original downloader or transport. |
| Prior existence | Both timestamps precede [Issue #5](https://github.com/xinnan369/looped-transform/issues/5) and its download approval rules by almost one year. LC-002R found the file already present on the dataset volume and read its ZIP metadata and hash. |
| Approval status | 3,722,384,133 bytes exceeds the 1 GB preapproval threshold for any new transfer. An original authorization record was not found. The matching release digest establishes file identity only. The original mirror and VPN route are unverified. |

This shard covers 1,000 of 5,109 originals. Further NWPU data use and acquisition are paused. Full-image audit remains blocked and official F1 is `not_ready`. No files were downloaded to establish this audit.
