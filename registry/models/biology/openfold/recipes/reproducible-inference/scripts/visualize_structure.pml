# Reproduce the OpenFold 6KWC visualizations.
# Run from the reproducible-inference directory:
# pymol -cq scripts/visualize_structure.pml

# Prediction colored by pLDDT
load results/6KWC_1_model_1_ptm_unrelaxed.pdb, openfold

hide everything
show cartoon, openfold
spectrum b, red_yellow_green, openfold, minimum=50, maximum=100
bg_color white
orient openfold
set ray_opaque_background, off
ray 1600, 1200
png visualizations/6KWC_1_openfold_plddt.png

# Prediction vs. experimental structure
delete all

load results/6KWC_1_model_1_ptm_aligned_to_6KWC.pdb, openfold
load reference/6KWC.pdb, experimental

hide everything
show cartoon, openfold
show cartoon, experimental and chain A
color marine, openfold
color gray70, experimental and chain A

bg_color white
orient openfold or (experimental and chain A)
set ray_opaque_background, off
ray 1600, 1200
png visualizations/6KWC_1_openfold_vs_experimental.png

quit
