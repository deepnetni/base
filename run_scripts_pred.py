import os
from utils.logger import CPrint

cp = CPrint()

epoch = 70

cmd = "python train_mcse_chime.py --vtest --pred"
cmd_valid = "python scripts/compute_metrics.py"

src_test = "~/datasets/CHiME3/data/audio/16kHz/isolated/test"
src_valid = "~/datasets/CHiME3/data/audio/16kHz/isolated/dev"
ckpath = "trained_mcse_chime3"

# python train_spatial_chime.py --pred --vtest --ckpt trained_mcse_chime3/spatial/checkpoints/epoch_0100.pth --out trained_mcse_chime3/pred_spatial_vtest_100

ckpt_file = f"epoch_0{epoch}.pth" if epoch == 100 else f"epoch_00{epoch}.pth"

elements = [
    # ("mcse_3en_wdefu_wfafu_c72_B2", f"{ckpath}/pred_mcse_vtest_{epoch}"),
    # ("mcse_baseline", f"{ckpath}/pred_baseline_vtest_{epoch}"),
    # ("mcse_baseline+3en", f"{ckpath}/pred_baseline_3en_vtest_{epoch}"),
    # ("mcse_baseline+3aff", f"{ckpath}/pred_baseline_3aff_vtest_{epoch}"),
    # ("mcse_baseline+3att", f"{ckpath}/pred_baseline_3att_{epoch}"),
    # ("mcse_baseline+UC+ED", f"{ckpath}/pred_baseline_UC+ED_vtest_{epoch}"),
    # ("mcse_baseline+3en+ED", f"{ckpath}/pred_baseline+3en+ED_vtest_{epoch}"),
    # ("mcse_uc_wdefu_wfafu_c72_B2", f"{ckpath}/pred_UC+all_vtest_{epoch}"),
    # ("mcse_3en_all_aff_wdefu_wfafu_c72_B2", f"{ckpath}/pred_3AFF+all_vtest_{epoch}"),
    # ("mcse_3en_all_att_wdefu_wfafu_c72_B2", f"{ckpath}/pred_3ATT+all_vtest_{epoch}"),
    # ("mcse_3en_wdefu_wfafu_inv_c72_B2", f"{ckpath}/pred_inv_all_vtest_{epoch}"),
    # ("mcse_baseline+UC+ED", f"{ckpath}/pred_baseline_uc_ed_vtest_{epoch}"),
    ("mcse_3en_wdefu_wfafu_cau_c72_B2", f"{ckpath}/pred_mcse_3en_wdefu_wfafu_cau_vtest_{epoch}"),
    #
    # ("mcnet", f"{ckpath}/pred_mcnet_vtest_100"),
    # ("deftan", f"{ckpath}/pred_deftan_vtest_100"),
    # ("spatial", f"{ckpath}/pred_spatial_vtest_100"),
]

for name, outdir in elements:
    # name = "mcse_3en_wdefu_wfafu_c72_B2"
    # outdir = f"{ckpath}/pred_mcse_vtest_{epoch}"
    cp.r(name)

    os.system(
        f"{cmd} --name {name} --ckpt {ckpath}/{name}/checkpoints/{ckpt_file} --out {outdir}"
    ) if (not os.path.exists(outdir) and name not in ["mcnet", "deftan", "spatial"]) else None
    # os.system(f"{cmd_valid} --sub --out {outdir}  --src {src_test}  --metrics asr pesqn pesqw stoi")
    os.system(
        f"{cmd_valid} --sub --out {outdir}  --excel {ckpath}/{name}_{epoch}.xlsx --src {src_test}  --metrics asr pesqn pesqw stoi"
    )
