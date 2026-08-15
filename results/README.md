| Date       | Commit                                   | Hardware                                                    | Z3 Version |
| ---------- | ---------------------------------------- | ----------------------------------------------------------- | ---------- |
| 2026-08-12 | e86ef2d2e36245b657866ee5de480c9a6a052931 | 128-core 2.25 GHz AMD EPYC 7742 processor and 995 GB of RAM | 4.8.12     |
| 2026-08-14 | cc9fb4344e6cc839e2423746274ce7b805f43715 | 128-core 2.25 GHz AMD EPYC 7742 processor and 995 GB of RAM | n/a        |

The 2026-08-14 run is `generate category reduction` only (no solve or Z3
phases): it re-measures the reduction rows after Red-5/6/7 were recompiled
with BLOCKSIZE=128 to match Red-1--4, and supersedes the 2026-08-12 numbers
for those three racy kernels. All other 2026-08-12 results remain current.
