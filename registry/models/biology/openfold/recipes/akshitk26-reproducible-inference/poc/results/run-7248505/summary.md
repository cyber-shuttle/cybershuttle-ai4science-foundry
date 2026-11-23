# proof-of-concept summary: run-7248505

| target | condition | reference | MSA seqs | mean pLDDT | CA pLDDT>70 | CA RMSD (Å) | TM-score | identity | matched CA | inference (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 6KWC_A | OpenFold example MSA | 6KWC:A | 5006 | 96.67 | 100% | 1.152 | 0.975 | 100.00% | 191/191 | 26.3 |
| 1AKE_A | ColabFold MSA | 1AKE:A | 16866 | 92.57 | 100% | 1.684 | 0.928 | 100.00% | 214/214 | 30.2 |
| 1AKE_A | ColabFold MSA | 4AKE:A | 16866 | 92.57 | 100% | 5.981 | 0.719 | 100.00% | 214/214 | 30.2 |
| 2LZM_A | ColabFold MSA | 2LZM:A | 4079 | 96.18 | 100% | 1.905 | 0.873 | 100.00% | 164/164 | 20.3 |
| 3CHY_A | ColabFold MSA | 3CHY:A | 11989 | 93.80 | 100% | 0.520 | 0.985 | 100.00% | 128/128 | 14.4 |
| 6KWC_A | ColabFold MSA | 6KWC:A | 5615 | 96.75 | 100% | 1.158 | 0.975 | 100.00% | 191/191 | 25.4 |
| 8TIM_A | ColabFold MSA | 8TIM:A | 16749 | 96.44 | 100% | 0.629 | 0.989 | 100.00% | 247/247 | 37.8 |
| 1AKE_A | single sequence | 1AKE:A | 1 | 47.10 | 3% | 20.455 | 0.252 | 100.00% | 214/214 | 30.0 |
| 1AKE_A | single sequence | 4AKE:A | 1 | 47.10 | 3% | 19.924 | 0.264 | 100.00% | 214/214 | 30.0 |
| 2LZM_A | single sequence | 2LZM:A | 1 | 53.50 | 21% | 16.175 | 0.313 | 100.00% | 164/164 | 20.4 |
| 3CHY_A | single sequence | 3CHY:A | 1 | 51.09 | 6% | 11.578 | 0.315 | 100.00% | 128/128 | 14.7 |
| 6KWC_A | single sequence | 6KWC:A | 1 | 47.77 | 8% | 21.157 | 0.214 | 100.00% | 191/191 | 25.5 |
| 8TIM_A | single sequence | 8TIM:A | 1 | 41.99 | 1% | 21.617 | 0.210 | 100.00% | 247/247 | 37.7 |

## MSA vs single sequence

| target | pLDDT with MSA | pLDDT single | RMSD with MSA (Å) | RMSD single (Å) |
|---|---|---|---|---|
| 6KWC_A | 96.75 | 47.77 | 1.158 | 21.157 |
| 3CHY_A | 93.80 | 51.09 | 0.520 | 11.578 |
| 2LZM_A | 96.18 | 53.50 | 1.905 | 16.175 |
| 1AKE_A | 92.57 | 47.10 | 1.684 | 20.455 |
| 8TIM_A | 96.44 | 41.99 | 0.629 | 21.617 |

## does mean pLDDT track accuracy?

Over 11 predictions (primary reference only): Pearson r(pLDDT, CA RMSD) = -0.97, Spearman rho = -0.84. Negative means higher confidence goes with lower error. With this few points, treat the sign and rough size as the result, not the exact value.

TM-score is computed for the fixed sequence pairing with the TM-score search heuristic, so it is a lower bound of the exact value. pLDDT is the atom average, the same number `scripts/validate_output.py` prints.
