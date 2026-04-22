# data

This directory stores local datasets and dataset caches.

Policy:
- Keep this directory in git.
- Do not track dataset files in git.
- Symlink datasets into this directory and name them according to `configs/dataset`.
- For LMDB caches, use `data/lmdb-cache` (recommended symlink target: `/root/autodl-tmp/lmdb-cache`).
