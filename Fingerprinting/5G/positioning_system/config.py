# =============================================================================
#  config.py – Konfiguration analog zum Matlab-Beispiel "NR Positioning Using PRS"
# =============================================================================
import numpy as np

MODE = "simulated"

# --- Carrier (analog: nrCarrierConfig) ---
CARRIER_FREQUENCY_HZ  = 3.0e9    # fc = 3e9 wie im Matlab-Beispiel
SCS_HZ                = 15e3     # SubcarrierSpacing = 15 kHz (Standard)
NUM_RBS               = 52       # NSizeGrid = 52
NSC                   = NUM_RBS * 12   # 624
NFFT                  = 1024
SLOTS_PER_FRAME       = 10       # bei 15 kHz SCS
N_FRAMES              = 1        # nFrames = 1
C                     = 3e8

# --- PRS (analog: nrPRSConfig) ---
PRS_RESOURCE_SET_PERIOD = [10, 0]  # PRSResourceSetPeriod
PRS_SLOT_OFFSETS        = list(range(0, 2*5, 2))  # 0:2:(2*numgNBs-1) → [0,2,4,6,8]
PRS_REPETITION          = 1
PRS_TIME_GAP            = 1
PRS_MUTING_1            = []
PRS_MUTING_2            = []
PRS_NUM_RBS             = 52      # NumRB
PRS_RB_OFFSET           = 0      # RBOffset
PRS_COMB_SIZE           = 12     # CombSize = 12 (wie Matlab)
PRS_NUM_SYMBOLS         = 12     # NumPRSSymbols = 12 (wie Matlab)
PRS_SYMBOL_START        = 0      # SymbolStart = 0

# --- Topologie ---
N_BS          = 5        # numgNBs = 5
N_UE          = 1
ISD_M         = 500.0    # Abstände wie in Matlab (~4000-9000m UMa → wir skalieren für Indoor)
BS_HEIGHT_M   = 25.0     # gNB height 25m (UMa, TR 38.901)
UE_HEIGHT_M   = 2.0      # UE height 2m

CELLS_TO_DETECT = 3      # cellsToBeDetected = min(3, numgNBs)

# --- UE-Position (analog: UEPos = [500 -20 2]) ---
UE_POS = np.array([500.0, -20.0, 2.0])

# --- Sender/Rauschen ---
PT_DBM          = 43
PT_W            = 10**(0.1*(PT_DBM-30))
NOISE_FIGURE_DB = 7
NOISE_POWER_W   = 1.380649e-23 * 290 * SCS_HZ * 10**(NOISE_FIGURE_DB/10)

# --- Kanalmodell ---
TDL_EXCESS_DELAYS_S = np.array([0,65,150,290,410,520,600,700,960,1190,1390,1530,1960,2350,2600])*1e-9
TDL_POWERS_DB       = np.array([-4.4,-1.2,-3.5,-5.2,-2.5,0.0,-2.2,-3.9,-7.4,-7.1,-10.7,-11.1,-5.1,-6.8,-8.7])
RNG_SEED            = 0   # rng('default') in Matlab

# --- ToA ---
TOA_IFFT_MULT = 16

# --- Visualisierung ---
VIZ_SAVE_PLOTS = True
VIZ_OUTPUT_DIR = "output"

# -----------------------------------------------------------------------------
#  Pathloss-Konfiguration (analog: nrPathLossConfig)
# -----------------------------------------------------------------------------
PATHLOSS_SCENARIO = "UMa"   # "UMa" oder "UMi" (wie plCfg.Scenario in Matlab)
PATHLOSS_LOS      = True    # losFlag = true (nur LOS wie im Matlab-Beispiel)
