import os
from utils.logger import CPrint

cp = CPrint()

epoch = 70

cmd = "python draw_mcse_atten_map.py"
ckpath = "trained_mcse_chime3"


elements = [
    ("mcse_3en_wdefu_wfafu_c72_B2", f"{ckpath}/draw_mcse_vtest_{epoch}"),
    # ("mcse_3en_all_att_wdefu_wfafu_c72_B2", f"{ckpath}/draw_mcse_vtest_{epoch}"),
    #
    # ("mcse_baseline", f"{ckpath}/pred_baseline_vtest_{epoch}"),
    # ("mcse_baseline+3en", f"{ckpath}/pred_baseline_3en_vtest_{epoch}"),
    # ("mcse_baseline+3aff", f"{ckpath}/pred_baseline_3aff_vtest_{epoch}"),
    # ("mcse_baseline+3att", f"{ckpath}/pred_baseline_3att_{epoch}"),
    # ("mcse_baseline+UC+ED", f"{ckpath}/pred_baseline_UC+ED_vtest_{epoch}"),
    # ("mcse_baseline+3en+ED", f"{ckpath}/pred_baseline+3en+ED_vtest_{epoch}"),
    # ("mcse_uc_wdefu_wfafu_c72_B2", f"{ckpath}/pred_UC+all_vtest_{epoch}"),
    # ("mcse_3en_all_aff_wdefu_wfafu_c72_B2", f"{ckpath}/pred_3AFF+all_vtest_{epoch}"),
    # ("mcse_3en_wdefu_wfafu_inv_c72_B2", f"{ckpath}/pred_inv_all_vtest_{epoch}"),
]

for name, outdir in elements:
    # name = "mcse_3en_wdefu_wfafu_c72_B2"
    # outdir = f"{ckpath}/pred_mcse_vtest_{epoch}"
    cp.r(name)

    os.system(
        f"{cmd} --name {name} --ckpt {ckpath}/{name}/checkpoints/epoch_00{epoch}.pth --out {outdir}"
    )
