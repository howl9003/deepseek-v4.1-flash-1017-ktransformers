# Contract v2 reference

`contract_v2.py` is the NumPy oracle for the integer-exact expert contract: E4M3 digits per 32, exact integer sums, one rounding. It checks the hashes of the FP8 and SiLU lookup tables (`fp8_table.bin`, `silu_table.bin`), which are deterministic numerical tables, not model data. `ORIGINAL_HASHES.json` pins its inputs.
