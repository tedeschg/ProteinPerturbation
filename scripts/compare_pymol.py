from pymol import cmd
import os
import glob

# =========================
# CONFIG
# =========================
REF_NAME = "/home/tedeschg/prj/side_projects/protein-perturbation/abl-imatinib_clean.pdb"
MUT_PATTERN = "*.pdb"

# =========================
# CHECK REFERENCE
# =========================
if not os.path.exists(REF_NAME):
    raise Exception(f"Reference not found: {REF_NAME}")

# =========================
# LOAD MUTANTS (current folder)
# =========================
cwd = os.getcwd()
mut_files = glob.glob(os.path.join(cwd, MUT_PATTERN))

# remove reference if accidentally included
mut_files = [f for f in mut_files if os.path.abspath(f) != os.path.abspath(REF_NAME)]

print(f"Reference: {REF_NAME}")
print(f"Mutants found: {len(mut_files)}")

# =========================
# LOAD STRUCTURES
# =========================
cmd.load(REF_NAME, "ref")

mut_names = []

for f in mut_files:
    name = os.path.splitext(os.path.basename(f))[0]
    cmd.load(f, name)
    mut_names.append(name)

# =========================
# ALIGN TO REFERENCE
# =========================
for m in mut_names:
    cmd.align(m, "ref")

# =========================
# VISUAL STYLE
# =========================
cmd.hide("everything")
cmd.show("cartoon", "all")
cmd.color("grey70", "all")

# =========================
# BUILD REFERENCE MAP
# =========================
ref_model = cmd.get_model("ref")
ref_map = {}

for a in ref_model.atom:
    ref_map[(a.chain, a.resi)] = a.resn

# =========================
# FIND MUTATIONS
# =========================
for m in mut_names:
    model = cmd.get_model(m)

    for a in model.atom:
        key = (a.chain, a.resi)

        if key in ref_map:
            if a.resn != ref_map[key]:
                cmd.color("red", f"{m} and chain {a.chain} and resi {a.resi}")
        else:
            # insertion / deletion
            cmd.color("orange", f"{m} and chain {a.chain} and resi {a.resi}")

# =========================
# FINAL VIEW
# =========================
cmd.bg_color("white")
cmd.orient()
cmd.set("cartoon_transparency", 0.1)

print("Done: mutation mapping completed")
