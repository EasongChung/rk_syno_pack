BEGIN_CONTEXT_PACKET
packet_id: npu-1x1-conv-244-jury-mini
requested_reasoner: muse

## Objective
Independent technical judgement on one narrow question. Answer in UNDER 250 words, plain Markdown.

## Evidence
Mesa etnaviv (Vivante NPU, NN core "v7") driver source: `etnaviv_ml_nn_v7.c`.
- :192-193 `cores_used = MIN2(out_channels, nn_core_count=8); kernels_per_core = DIV_ROUND_UP(out_channels, cores_used);`
- :220-222 last superblock uses `kernels_in_superblock = kernels_per_core - d*(superblocks-1)` (= rem).
- :225/:231/:305/:389 `out_channel = core*kernels_in_superblock + kernel + superblock*d*cores_used;`
- :226 weight read `input + out_channel*W*H*in_c`; :238 `biases[out_channel]`; :251 GPU output offset
  `append_bits(out_values_per_channel*out_channel, 32)` with out_values_per_channel = 2500.
- :541-546 path choice: 1x1 (pointwise, out>8) -> write_core_6; depthwise (in_c==1) -> write_core_sequential.

Because kernels_per_core rounds the channel count UP to a multiple of 8, the last superblock always
executes m = 8*ceil(C/8) - C non-existent "phantom" channels (m in 0..7), causing 3 out-of-bounds
accesses. Measured on synthetic 1x1 int8 convs (geometry 50x50):

C=122 m=6 no-hang | C=196 m=4 no-hang | C=220 m=4 no-hang | C=240 m=0 no-hang |
C=244 m=4 HANG | C=260 m=4 no-hang | C=280 m=0 no-hang | C=248 m=0 no-hang |
C=252 m=4 no-hang | C=284 m=4 HANG

For C=244 the last superblock has rem=1 and cores 4..7 are ENTIRELY phantom (no valid channel);
for C=252 rem=2 and only cores 6..7 are entirely phantom. For C=248 rem=1 but m=0 (no phantom at all).

## Constraints
- Answer only from the text above plus your own reasoning. Do not use tools.
- If something is unknowable, say "unknown".
- Keep it under 250 words.

## Questions
1. Does the phantom-channel out-of-bounds mechanism explain all three of: 244 hangs, 252 does not,
   248 does not? If not, what is the discriminating quantity?
2. Is the rule "hang iff rem==1 AND m>0" identifiable from only these 10 points? Name one alternative
   predicate that fits all 10 points equally well.
3. Rank two competing root causes for the hang and give one offline (no-device) test for each.
END_CONTEXT_PACKET
