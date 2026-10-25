import os

MODEL = "opus-4.6"

DIR_BASIC = "./Dataset/Basic"
DIR_ADVANCED = "./Dataset/Advanced"

EXP_BASIC = "./experiment-basic"
EXP_ADVANCED = "./experiment-advanced"

DATASETS_BASIC = [
    "BernardCells",
    "Cavity", 
    "counterFlowFlame2D",
    "Cylinder",
    "forwardStep",
    "obliqueShock", 
    "pitzDaily",
    "squareBend",
    "wedge",
    "shallowWaterWithSquareBump",
    "damBreakWithObstacle"
]
CASES_BASIC = [1, 2, 3,
        #  4, 5, 6, 7, 8, 9, 10
         ]

# Case-list override for held-out evaluation. The default above stays [1, 2, 3]
# so existing submissions are unaffected; set FOAMAGENT_CASES_BASIC="8 9 10"
# (or "8,9,10") in the job environment to benchmark other variants.
_cases_override = os.environ.get("FOAMAGENT_CASES_BASIC", "").strip()
if _cases_override:
    CASES_BASIC = [int(tok) for tok in _cases_override.replace(",", " ").split()]

DATASETS_ADVANCED = [
    "Cavity_LES",
    "Cavity_SA",
    "Cavity_geometry_1",
    "Cylinder_LES",
    "Cylinder_SA",
    "Diamond_Obstacle_KOMEGASST",
    "Diamond_Obstacle_SA",
    "Double_Square_SA",
    "Rectangular_Obstacle_KOMEGASST",
    "Rectangular_Obstacle_SA",
    "counterFlowFlame2D_KE",
    "counterFlowFlame2D_SA",
    "nozzleFlow2D_SA",
    "obliqueShock_KE",
    "obliqueShock_LES",
    "wedge_SA"
]