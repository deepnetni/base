import os
from utils.logger import CPrint

cp = CPrint()

epoch = 100
vt = "valid"
# vt = "valid"


cmd = f"python train_mcse_l3das.py --{vt} --pred"
cmd_mcnet = f"python train_mcnet_l3das.py --{vt} --pred"
cmd_deftan = f"python train_deftan_l3das.py --{vt} --pred"

cmd_valid = "python scripts/compute_metrics.py"


src_test = "/home/deepni/datasets/l3das/L3DAS22_Task1_dev/labels"
src_valid = "/home/deepni/datasets/l3das/L3DAS22_Task1_dev/labels_val"
src_dir = src_test if vt == "vtest" else src_valid

ckpath = "trained_mcse_l3das_win"

ckpt_file = f"epoch_0{epoch}.pth" if epoch == 100 else f"epoch_00{epoch}.pth"


elements = [
    ("mcse_3en_wdefu_wfafu_c72_B2", f"{ckpath}/pred_mcse_{vt}_{epoch}"),
    ("mcse_baseline", f"{ckpath}/pred_baseline_{vt}_{epoch}"),
    # ("mcse_baseline+3en", f"{ckpath}/pred_baseline_3en_vtest_{epoch}"),
    # ("mcse_baseline+UC+ED", f"{ckpath}/pred_baseline_UC+ED_vtest_{epoch}"),
    #
    # ("mcse_baseline+3aff", f"{ckpath}/pred_baseline_3aff_vtest_{epoch}"),
    # ("mcse_baseline+3att", f"{ckpath}/pred_baseline_3att_{epoch}"),
    # ("mcse_baseline+3en+ED", f"{ckpath}/pred_baseline+3en+ED_vtest_{epoch}"),
    # ("mcse_uc_wdefu_wfafu_c72_B2", f"{ckpath}/pred_UC+all_vtest_{epoch}"),
    # ("mcse_3en_all_aff_wdefu_wfafu_c72_B2", f"{ckpath}/pred_3AFF+all_vtest_{epoch}"),
    # ("mcse_3en_all_att_wdefu_wfafu_c72_B2", f"{ckpath}/pred_3ATT+all_vtest_{epoch}"),
    # ("mcse_3en_wdefu_wfafu_inv_c72_B2", f"{ckpath}/pred_inv_all_vtest_{epoch}"),
    #
    # ("mcnet_raw", f"{ckpath}/pred_mcnet_{vt}_{epoch}"),
    # ("deftan", f"{ckpath}/pred_deftan_{vt}_{epoch}"),
]

for name, outdir in elements:
    # name = "mcse_3en_wdefu_wfafu_c72_B2"
    # outdir = f"{ckpath}/pred_mcse_vtest_{epoch}"
    cp.r(name)

    if name == "mcnet_raw":
        os.system(
            f"{cmd_mcnet} --ckpt {ckpath}/{name}/checkpoints/{ckpt_file} --out {outdir}"
        ) if not os.path.exists(outdir) else None
    elif name == "deftan":
        os.system(
            f"{cmd_deftan} --ckpt {ckpath}/{name}/checkpoints/{ckpt_file} --out {outdir}"
        ) if not os.path.exists(outdir) else None
    else:
        os.system(
            f"{cmd} --name {name} --ckpt {ckpath}/{name}/checkpoints/{ckpt_file} --out {outdir}"
        ) if not os.path.exists(outdir) else None
    # os.system(f"{cmd_valid} --sub --out {outdir}  --src {src_test}  --metrics asr pesqn pesqw stoi")
    os.system(
        f"{cmd_valid} --sub --out {outdir}  --excel {name}_{vt}_{epoch}.xlsx --src {src_dir}  --metrics asr pesqn pesqw stoi"
    )
