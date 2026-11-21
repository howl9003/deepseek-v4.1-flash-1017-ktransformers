# DeepSeek V4.1 Flash 1017 · C4 r125

C4 serves the 1017 checkpoint on two 16 GB RTX 5070 Ti GPUs with eight CPU NUMA parts. It uses Contract v3 integer expert accumulation, a prebuilt CPU expert image and plain decode. Table A, the equal split, is the default. The release adds atomic diagnostic writes to the measured build and makes machine-specific paths configurable. The software licence is Apache-2.0.

## r120 results on the test server

These measurements were recorded on r120. Qualification on the rebased r125 trees is pending.

The official regrade passed all eight gates covering identity, image loading, thermal windows, speed, cold prefill traces, switch activation, the reference panel and table A.

| Measurement | Result |
|---|---|
| Identity against both preceding plain references | 64/64 logits hash comparisons, 256/256 full logprob values, 12/12 greedy outputs, zero two- or four-request hash differences |
| 1017 reference panel | 199 conversations, 254,846 positions. Mean KL 0.00978253 nats, served NLL 0.47020820, top-1 agreement 97.0323%. Paired mean KL and NLL differences were −3.69 × 10⁻⁸ and +1.49 × 10⁻⁸ nats. Only 2/199 capture files were byte-identical. |
| Table B throughput at 1 / 2 / 4 sessions | **47.43 / 66.79 / 76.17 aggregate tokens/s**, means of three 512-token runs. Six-workload breadth averaged 48.10 tokens/s. |
| Table B cold 12,288 / 35,000 tokens | **9.45 / 21.03 s**, mean client wall times. |
| Table B weight load | **122.68 s**. All 40 expert layers used direct image reads. All 320 part digests matched. Prebuilt extensions required zero builds at server load. |
| Table A, measured | **48.42 tokens/s** at one session, median across 60 × 2,048-token rounds in 2,709.3 s. Three P2 memory thermal assertions at 85 °C. |
| Table B, measured | **74.47 aggregate tokens/s** over a four-session soak lasting 2,753.1 s. Zero P2 assertions or overlapping thermal windows. |

The table A short-run **estimates** are 50.0 / 69.6 / 83.7 aggregate tokens/s before throttling. Sustained table A rates at two and four sessions were unmeasured. Use table B temporarily when DIMM temperatures constrain the equal split.

## Rebuild and serve

Use these release branches.

| Checkout | Repository | Branch |
|---|---|---|
| SGLang | howl9003/sglang | `main` |
| Recipe | howl9003/deepseek-v4.1-flash-1017-ktransformers | `main` |
| kt overlay | howl9003/deepseek-v4.1-flash-1017-ktransformers | `c4-r120-kt` |

The SGLang tree is `f62b8d11d5d10a30b53b7f836283ee4fd85aa4f0` and the kt overlay tree is `36f72b46dccf6ce3036f2c6162625d3adf147bf9`. Create the public base installation, build the native kernel and image, and qualify the GPU library on each device. The recipe book has the Linux commands and lists the external inputs. [Rebuild and calibration](RECIPE.md)

Copy `config.example.json` to a local file, replace the paths and insert the two real GPU UUIDs in rank order. Launch from the recipe directory after preparing the installation.

```sh
python3 -I -B preflight.py "$C4_ROOT/tree/third_party/sglang" "$KT" --config config.local.json
python3 -I -B launch.py plan config.local.json
python3 -I -B launch.py prebuild config.local.json
python3 -I -B launch.py serve config.local.json > table-A.log 2>&1
```

For table B, stop the server and relaunch with `--table B`. Table A uses 288 rows per part. Table B uses 320,320,320,320,192,192,320,320. Both profiles keep speculative decoding off.

## Check identity, image loading and performance

Run these probes against a dedicated server. The greedy probes flush its prefix cache. The default endpoint is `http://127.0.0.1:18209`. Clients accept another endpoint through `C4_URL`.

```sh
python3 -I -B gates/check_image.py table-A.log
python3 -I -B gates/check_identity.py
python3 -I -B gates/measure.py
python3 -I -B launch.py verify config.local.json
```

The identity command compares both paths to the frozen logits and logprob values, checks the 20K lane, and repeats paired and four-request batches. The public JSON prompt uses redacted example contacts. It is checked across modes. The other nine greedy outputs are checked against the frozen run. Supply the original fixture with `--json-prompt FILE` for all twelve frozen comparisons. The tool checks the fixture digest before sending it.

## Score the reference panel

Supply the tokenized 1017 panel and the matching reference top-128 safetensor parts in **sequence order**. The capture checks the panel digest, keeps the served top 256 and writes a new directory. Run the scorer with PyTorch and safetensors installed. Set `C4_PAIRED_CAPTURE` to the qualified baseline capture directory to also enforce the paired 10⁻⁷-nat gates.

```sh
python3 -I -B gates/capture_panel.py "$PANEL" capture
python3 -I -B gates/score_panel.py capture scores.json gates/panel_index.json "$PARTS"/part-*.safetensors
```

The scorer uses the reference top-128 support plus one tail bucket. Each native ID missing from the served top 256 receives the smallest kept probability. The three model chapters explain the measurements and the limits of the fitted constants. [Decode](DECODE_MODEL_r1.md) · [Prefill](PREFILL_MODEL_r1.md) · [Placement](PLACEMENT_MODEL_r1.md)
