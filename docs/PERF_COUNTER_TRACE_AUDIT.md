# `perf_counter` trace audit

This audit covers every production Python file containing `perf_counter` in
`modelexpress_client/python` (tests and benchmarks have their own clocks). The
counts include both start and end reads. Re-run
`rg -n 'perf_counter\(' modelexpress_client/python --glob '*.py'` after edits to
locate each call. A clock used to enforce a timeout, correlate NIXL completion,
or preserve an existing public metrics record must remain a clock; an OpenTelemetry
span cannot replace that value.

| File under `modelexpress_client/python/` | Reads at audit | Trace decision |
| --- | ---: | --- |
| `modelexpress/refit/timing.py` | 1 | Clock dependency for the compatibility JSON recorder. Its `span()` now also emits an OTel span named for `duration_key` when present. |
| `modelexpress_rl/train/runtime.py` | 1 | Compatibility timer callback; the MX publication cycle has an OTel span. |
| `modelexpress_rl/train/engines/fsdp/publisher.py` | 4 | Manifest generation and serialization now have spans. Clock reads remain for the optional legacy `metrics` argument. |
| `modelexpress_rl/inference/nixl_staged_transfer.py` | 49 | Source discovery and cache lookup, decode, merge, source build, layout capture, plan cache lookup and validation, planning, registration, wire post/wait, and reconstruction have spans. Clocks still populate internal transfer records and calculate the throughput floor. |
| `modelexpress_rl/inference/engines/vllm/installer.py` | 14 | Streaming install, commit, retention scans, reload and post-install synchronization have spans. Clocks still populate the artifact's private transfer record. |
| `modelexpress_rl/inference/engines/vllm/direct_copy.py` | 4 | Direct copy commit batches have spans. Clocks still populate the artifact's private transfer record. |
| `modelexpress_rl/inference/engines/vllm/direct_glm.py` | 4 | GLM guard and derived refresh have spans. Clocks still populate the artifact's private transfer record. |
| `modelexpress_rl/inference/methods/load_time_tensor.py` | 2 | Load-time tensor materialization is outside a timed refit; its clock feeds the existing artifact diagnostics. |
| `modelexpress_rl/inference/receiver.py` | 10 | S3 download, validation and delta application are span candidates for the checkpoint path; their seconds are still needed by the existing checkpoint result. |
| `modelexpress_rl/inference/engines/vllm/weight_transfer_engine.py` | 2 | Converted to stage and apply spans; no metric-dictionary merge. |
| `modelexpress/refit/reshard/receiver.py` | 39 | Legacy direct reshard prepare, wire and install stages are span candidates; the existing stage map still drives its diagnostic record and throughput checks. |
| `modelexpress/refit/reshard/rendezvous.py` | 6 | Fetch/parse phase durations are span candidates; accumulated totals still feed diagnostics. |
| `modelexpress/nixl_transfer.py` | 31 | Timeout, posted/submitted/completed timestamps and throughput arithmetic require clocks; allocation, registration and transfer setup intervals are span candidates. |
| `modelexpress/load_strategy/rdma_strategy.py` | 12 | Selection, fetch and transfer intervals are span candidates; fallback latency values remain part of existing load results. |
| `modelexpress/metadata/artifact_transfer.py` | 11 | Artifact validation, extraction and transfer intervals are span candidates; current duration values are exposed in results/logs. |
| `modelexpress/metadata/artifact_lifecycle.py` | 4 | Artifact lifecycle intervals are span candidates; current duration values are exposed in logs. |
| `modelexpress/metrics.py` | 9 | Existing Prometheus timing wrappers and metric observations need clock values; replacing those clocks would drop existing metrics. |
| `modelexpress/engines/vllm/refit/installer.py` | 4 | Generic vLLM install spans are candidates; the existing measured result is still used in diagnostics. |
| `modelexpress/engines/vllm/loader.py` | 2 | Model-load span candidate; not part of a timed refit. |
| `modelexpress/engines/sglang/loader.py` | 4 | Model-load span candidates; not part of a timed refit. |
| `modelexpress/engines/trtllm/loader.py` | 2 | Model-load span candidate; not part of a timed refit. |
| `modelexpress/gds_loader.py` | 2 | Model-load span candidate; not part of a timed refit. |
| `modelexpress/gds_transfer.py` | 2 | Timeout clock; cannot be replaced by a span. |
| `modelexpress/tensor_utils.py` | 2 | Tensor copy duration candidate; current elapsed value is exposed in diagnostics. |

## GLM breakdown CSV

The historical file `MX+RL Benchmark  data - GLM_2026_09_29_Breakdown.csv`
contains 173 rows. Rows with `kind=derived residual` or `kind=derived rate`
are calculations from other fields, not independent observations. The measured
`source_field` values fall into these origins:

| Origin | Current OTel representation |
| --- | --- |
| `trainer.elapsed_s`, `orchestrator.elapsed_s`, `generator.elapsed_s`, `*.phases_s.*` | PrimeRL MX root and phase spans, with `role`, `rank`, `step`, `version_uid`, experiment and staging mode. |
| Trainer `marks.*` from handshake, publication, rendezvous and release | PrimeRL MX child spans. Numerical poll and mode flags are root span attributes. |
| Trainer `marks.*` from MX source preparation and publication | MX refit spans named by their `duration_key`, parented to PrimeRL spans through in-process context; MX RPCs continue the trace on the server. |
| Generator `marks.streaming_*` and MX transfer internals | MX streaming prepare, apply and release spans and nested source, plan, wire, reconstruction and install spans. |
| `trainer-metrics.time/*` | PrimeRL's framework metrics. These are outside the MX transport and are not yet emitted through MX's OTel facade. |
| Historical GLM direct-only cache and derived-refresh fields | Source and plan cache lookups, direct install, guard, commit, and derived refresh now have spans on the GLM path. A Qwen bounded run will not emit GLM-only operations. |
| Transfer size, batch and rank fields | Numeric attributes on the active transfer/refit span. |

Duration fields should be read from span start/end timestamps. The OTel
histograms at MX/PrimeRL refit phase boundaries provide aggregate latency
metrics. No duration is also sent through a `telemetry.measurement` call.

For the CSV's updates 2–11, the framework broadcast timer exceeds the MX
trainer root by 9.1–20.1 ms per update (about 10 ms on nine of ten updates).
The transfer wrapper contributes 0.4–0.5 ms of that gap. The other framework
timings remain available as PrimeRL metrics without editing its generic
trainer path.
