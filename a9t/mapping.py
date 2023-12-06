TS_GYNE_MAP = {
    820: {
        "name": "TSGyneOrgans",
        "config": "3d_lowres",
        "map": {
            1: "Bladder",
            2: "Duodenum",
            3: "Pancreas",
            4: "Rectum",
            5: "Colon_Sigmoid",
            6: "Vs_Common_Iliac",
            7: "Vagina",
            8: "V_Venacava_I",
            9: "Stomach",
            10: "Spleen",
            11: "Liver",
            12: "Kidney_R",
            13: "Kidney_L",
            14: "Femur_R",
            15: "Femur_L",
            16: "Bone_Marrow_Pel",
            17: "Bone_Marrow_Lum",
            18: "Anus",
            19: "A_Aorta",
            20: "Uterus",
        },
        "trainer_name": "nnUNetTrainer",
        "postprocess": None,
    },
    821: {
        "name": "TSGyneCtvnExtIliac",
        "config": "3d_lowres",
        "map": {
            1: "Ctvn_Ext_Iliac",
        },
        "trainer_name": "nnUNetTrainer",
        "postprocess": None,
    },
    822: {
        "name": "TSGyneCtvnPresacral",
        "config": "3d_lowres",
        "map": {
            1: "Ctvn_Presacral",
        },
        "trainer_name": "nnUNetTrainer",
        "postprocess": None,
    },
    823: {
        "name": "TSGyneBowel",
        "config": "3d_lowres",
        "map": {
            1: "Bowel",
        },
        "trainer_name": "nnUNetTrainer",
        "postprocess": None,
    },
}

TS_PRIME_MAP = {
    # 720: {
    #     "name": "TSPrime",
    #     "config": "3d_fullres",
    #     "map": {
    #         1: "Bladder",
    #         2: "Anorectum",
    #         3: "Bag_Bowel",
    #         4: "Femur_Head_L",
    #         5: "Femur_Head_R",
    #         6: "Penilebulb",
    #     },
    #     "trainer_name": "nnUNetTrainerNoMirroring",
    #     "postprocess": "data/nnUNet_results/Dataset720_TSPrime/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/postprocessing.pkl",
    # },
    # 721: {
    #     "name": "TSPrimeCTVP",
    #     "config": "3d_lowres",
    #     "map": {
    #         1: "Ctvp",
    #     },
    #     "trainer_name": "nnUNetTrainer",
    #     "postprocess": None,
    # },
    722: {
        "name": "TSPrimeCTVN",
        "config": "3d_fullres",
        "map": {
            1: "Ctvn",
        },
        "trainer_name": "nnUNetTrainer",
        "postprocess": None,
    },
}

ALL_SEG_MAP = {
    "TSPrime": TS_PRIME_MAP,
    "TSGyne": TS_GYNE_MAP,
}
