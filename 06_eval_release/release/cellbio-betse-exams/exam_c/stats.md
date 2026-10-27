| system | mean reward over 20 batches |
|---|---|
| agent_k5 | 0.852 |
| nn1 | 0.827 |
| mixed_k5 | 0.807 |
| random_k0 | 0.772 |
| constant | 0.708 |
| mixed_k0 | 0.678 |
| agent_k0 | 0.602 |
| grid_k0 | 0.500 |

| comparison | diff | 95% CI | Wilcoxon p | wins a/b |
|---|---|---|---|---|
| **mixed_k5 vs agent_k5** | -0.045 | [-0.065, -0.025] | 0.0014 | 3/17 |
| grid_k0 vs constant | -0.208 | [-0.257, -0.159] | 0.0000 | 0/20 |
| random_k0 vs constant | +0.063 | [+0.033, +0.095] | 0.0014 | 16/4 |
| mixed_k0 vs constant | -0.030 | [-0.056, -0.005] | 0.0532 | 8/12 |
| mixed_k5 vs constant | +0.099 | [+0.053, +0.144] | 0.0006 | 18/2 |
| agent_k0 vs constant | -0.106 | [-0.143, -0.068] | 0.0001 | 2/18 |
| agent_k5 vs constant | +0.144 | [+0.109, +0.180] | 0.0000 | 20/0 |
| mixed_k5 vs mixed_k0 | +0.129 | [+0.096, +0.160] | 0.0000 | 19/1 |
| agent_k5 vs agent_k0 | +0.250 | [+0.227, +0.273] | 0.0000 | 20/0 |
| mixed_k5 vs random_k0 | +0.035 | [+0.011, +0.059] | 0.0153 | 14/6 |
| agent_k5 vs nn1 | +0.025 | [+0.012, +0.038] | 0.0033 | 17/2 |
| mixed_k5 vs nn1 | -0.020 | [-0.035, -0.006] | 0.0153 | 5/15 |
