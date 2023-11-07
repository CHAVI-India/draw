TS_GYNE_MAP = {
    1: "Body",
    2: "Bladder",
    3: "Bag_Bowel",
    4: "SpinalCord",
    5: "Duodenum",
    6: "Pancreas",
    7: "Rectum",
    8: "Colon_Sigmoid",
    9: "Vs_Int_Iliac",
    10: "Vs_Common_Iliac",
    11: "Vs_Ext_Iliac",
    12: "Vagina",
    13: "V_Venacava_I",
    14: "Stomach",
    15: "Spleen",
    16: "Liver",
    17: "Kidney_R",
    18: "Kidney_L",
    19: "Femur_R",
    20: "Femur_L",
    21: "Ctvn_Presacral",
    22: "Ctvn_Paraaortic",
    23: "Ctvn_Obturator",
    24: "Ctvn_Int_Iliac",
    25: "Ctvn_Ext_Iliac",
    26: "Ctvn_Com_Iliac",
    27: "Bowel",
    28: "Bone_Marrow_Pel",
    29: "Bone_Marrow_Lum",
    30: "Anus",
    31: "A_Aorta",
}

TS_PRIME_MAP = {
    720: {
        "name": "TSPrime",
        "config": "3d_fullres",
        "map": {
            1: "Bladder",
            2: "Anorectum",
            3: "Bag_Bowel",
            4: "Femur_Head_L",
            5: "Femur_Head_R",
            6: "Penilebulb",
        },
        "trainer_name": "nnUNetTrainerNoMirroring",
        "postprocess": "data/nnUNet_results/Dataset720_TSPrime/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/postprocessing.pkl",
    },
    721: {
        "name": "TSPrimeCTVP",
        "config": "3d_lowres",
        "map": {
            1: "Ctvp",
        },
        "trainer_name": "nnUNetTrainer",
        "postprocess": None,
    },
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
