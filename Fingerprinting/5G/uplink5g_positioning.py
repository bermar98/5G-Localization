# =============================================================================
#  5G Uplink Positioning Simulation
#  Method: TDoA (Time Difference of Arrival) with Least Squares Estimation
# =============================================================================

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import os
import numpy as np
import numpy.matlib
import scipy as sp
import scipy.io as spio
import scipy.constants
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

import sys
sys.path.append("../../../")

from toolkit5G.ChannelModels      import AntennaArrays, SimulationLayout, ParameterGenerator, ChannelGenerator
from toolkit5G.ResourceMapping    import ResourceMapperSRS
from toolkit5G.ReceiverAlgorithms import ChannelEstimationSRS
from toolkit5G.Positioning        import ToAEstimation, PositionEstimation, LeastSquareTDoA, LeastSquareToA
from toolkit5G.ChannelProcessing  import AddNoise


# =============================================================================
#  Simulationsparameter
# =============================================================================

propTerrain      = "UMi"           # Ausbreitungsszenario (Urban Micro)
carrierFrequency = 3.6e9           # Trägerfrequenz in Hz
scs              = 30e3            # Subcarrier-Abstand in Hz
Nfft             = 4096            # FFT-Größe
numOfBSs         = 21              # Anzahl Basisstationen
nBSs             = np.prod(numOfBSs)
nUEs             = 128             # Anzahl UEs
numRBs           = 272             # Anzahl Resource Blocks
numSlots         = 1


# =============================================================================
#  SRS-Konfiguration (Sounding Reference Signal)
# =============================================================================

purpose                      = "positioning"
nrofSRS_Ports                = 1
transmissionComb             = 4
nrofSymbols                  = 12
startPosition                = 2
repetitionFactor             = 1
nrOfCyclicShift              = 1
groupOrSequenceHopping       = "neither"
sequenceId                   = np.arange(nUEs)
systemFrameNumber            = 0
resourceType                 = "periodic"
subcarrierSpacing            = scs
bSRS                         = 0
cSRS                         = 61
bHop                         = 0
freqScalingFactor            = 1
startRBIndex                 = 0
enableStartRBHopping         = False
freqDomainShift              = 0
freqDomainPosition           = 0
srsPeriodicityInSlots        = 1
srsOffsetInSlots             = 0
betaSRS                      = 1
resourceGridSizeinRBs        = numRBs
Bandwidth                    = resourceGridSizeinRBs * 12 * scs


# =============================================================================
#  Antennen-Arrays
# =============================================================================

# UE-seitige Antenne: OMNI, 2 Panels, 2 Elemente pro Panel
ueAntArray = AntennaArrays(
    antennaType     = "OMNI",
    centerFrequency = carrierFrequency,
    arrayStructure  = np.array([1, 1, 2, 2, 1])
)
ueAntArray()

# BS-seitige Antenne: 3GPP_38.901, 4 Panels, 4 Elemente pro Panel
bsAntArray = AntennaArrays(
    antennaType     = "3GPP_38.901",
    centerFrequency = carrierFrequency,
    arrayStructure  = np.array([1, 1, 8, 4, 1])
)
bsAntArray()


# =============================================================================
#  Layout-Parameter
# =============================================================================

isd                  = 100          # Inter-Site-Distanz in m
minDist              = 10           # Min. Abstand UE–BS in m
ueHt                 = 1.5          # UE-Höhe in m
bsHt                 = 10           # BS-Höhe in m
bslayoutType         = "Hexagonal"  # BS-Layout-Typ
ueDropType           = "Hexagonal"  # UE-Verteilungstyp
htDist               = "equal"      # UE-Höhenverteilung
ueDist               = "random"     # UE-Verteilung pro Site
nSectorsPerSite      = 3            # Sektoren pro Site
maxNumFloors         = 1
minNumFloors         = 1
heightOfRoom         = 5.1          # Deckenhöhe in m
indoorUEfract        = 0.5          # Anteil Indoor-UEs
lengthOfIndoorObject = 3
widthOfIndoorObject  = 3
forceLOS             = True         # Alle Links im LOS-Zustand erzwingen


# =============================================================================
#  Simulations-Layout
# =============================================================================

simLayoutObj = SimulationLayout(
    numOfBS              = numOfBSs,
    numOfUE              = nUEs,
    heightOfBS           = bsHt,
    heightOfUE           = ueHt,
    ISD                  = isd,
    layoutType           = bslayoutType,
    layoutWidth          = 30,
    layoutLength         = 120,
    ueDropMethod         = ueDropType,
    UEdistibution        = ueDist,
    UEheightDistribution = htDist,
    numOfSectorsPerSite  = nSectorsPerSite,
    ueRoute              = None
)

simLayoutObj(
    terrain              = propTerrain,
    carrierFreq          = carrierFrequency,
    ueAntennaArray       = ueAntArray,
    bsAntennaArray       = bsAntArray,
    indoorUEfraction     = indoorUEfract,
    heightOfRoom         = heightOfRoom,
    lengthOfIndoorObject = lengthOfIndoorObject,
    widthOfIndoorObject  = widthOfIndoorObject,
    forceLOS             = forceLOS
)

# Topologie anzeigen
fig, ax = simLayoutObj.display2DTopology()
ax.set_xlabel("x-Koordinaten (m)")
ax.set_ylabel("y-Koordinaten (m)")
ax.set_title("Simulations-Topologie")

# Kanalparameter & Kanalkoeffizienten generieren
paramGen = simLayoutObj.getParameterGenerator()
print("Kanalparameter generiert!")

channel = paramGen.getChannel()
print("Kanalkoeffizienten generiert!")

Hf = channel.ofdm(scs, Nfft)[0]
print("OFDM-Kanal generiert!")

Nt = bsAntArray.numAntennas   # Anzahl BS-Antennen
Nr = ueAntArray.numAntennas   # Anzahl UE-Antennen


# =============================================================================
#  Transmission Grid & ToA-Schätzung
# =============================================================================

print("***********  Transmission Grid Beamformed *********** ")

numRepetition    = 1
numSlotsPerFrame = np.int32(10 * (15000 / scs))
numUEsPerSlot    = transmissionComb
numSlots         = np.int32(np.ceil(nUEs * numRepetition / transmissionComb))
frameIndices     = np.int32(np.floor(np.arange(numUEsPerSlot * numRepetition) / transmissionComb) % numSlotsPerFrame)
slotIndices      = np.int32(np.floor(np.floor(np.arange(numUEsPerSlot * numRepetition) / transmissionComb) / numSlotsPerFrame))
combOffset       = np.int32(np.arange(numUEsPerSlot))

ToAe = np.zeros((nBSs, nUEs))

for ns in range(numSlots):

    # --- SRS Grid generieren ---
    srsGrid   = np.zeros((numUEsPerSlot, 14, numRBs * 12), dtype=np.complex64)
    srsObject = np.empty((numUEsPerSlot), dtype=object)

    for nue in range(numUEsPerSlot):
        srsObject[nue] = ResourceMapperSRS(
            nrofSRS_Ports, transmissionComb, nrofSymbols, startPosition,
            repetitionFactor, nrOfCyclicShift, groupOrSequenceHopping,
            sequenceId[nue], combOffset[nue], ns, frameIndices[nue],
            resourceType, purpose, subcarrierSpacing
        )
        srsGrid[nue] = srsObject[nue](
            bSRS, cSRS, bHop, freqScalingFactor, startRBIndex,
            enableStartRBHopping, freqDomainShift, freqDomainPosition,
            srsPeriodicityInSlots, srsOffsetInSlots, betaSRS,
            resourceGridSizeinRBs
        )[0, 0, 0]

    XGrid     = np.zeros((numUEsPerSlot, 14, Nfft), dtype=np.complex64)
    bwpOffset = np.random.randint(Nfft - resourceGridSizeinRBs * 12)

    print("***********  SRS Grid generiert *********** ")

    # Ressourcengitter in Transmission Grid laden
    XGrid[..., bwpOffset:(bwpOffset + resourceGridSizeinRBs * 12)] = srsGrid
    print("***********  Transmission Grid generiert *********** ")
    del srsGrid

    # --- Beamforming ---
    Pt_dBm = 23
    Pt     = 10 ** (0.1 * (Pt_dBm - 30))
    lamda  = 3e8 / carrierFrequency
    d      = 0.5 / lamda
    theta  = 0
    Xf     = (transmissionComb * Pt / Nr) * XGrid[..., np.newaxis].repeat(Nr, axis=-1)
    del XGrid

    ueIndices = np.arange(ns * numUEsPerSlot, (ns + 1) * numUEsPerSlot)

    # --- Durch den Kanal senden ---
    Yf = (Hf[:, :, ueIndices].transpose(1, 2, 0, 3, 5, 4) @ Xf[np.newaxis, ..., np.newaxis]).sum(1)
    print(f"***********  [{ns}]-Durch Kanal gesendet *********** ")

    # --- Rauschen hinzufügen ---
    BoltzmanConst = 1.380649e-23
    temperature   = 300
    noisePower    = BoltzmanConst * temperature * scs
    Yf = np.complex64(
        Yf + np.sqrt(0.5 * noisePower) * (
            np.random.standard_normal(Yf.shape) + 1j * np.random.standard_normal(Yf.shape)
        )
    )
    print(f"***********  [{ns}]-Rauschen hinzugefügt *********** ")

    # --- Ressourcengitter extrahieren ---
    rxGrid = Yf[..., bwpOffset:(bwpOffset + resourceGridSizeinRBs * 12), :, 0].transpose(0, 3, 1, 2)
    print(f"***********  [{ns}]-Ressourcengitter extrahiert *********** ")

    # --- Kanalschätzung & Interpolation ---
    Hfest = np.zeros((nBSs, numUEsPerSlot, Nt, 14, rxGrid.shape[-1]), dtype=np.complex64)
    chEST = ChannelEstimationSRS()
    chGrid = rxGrid.reshape(nBSs * Nt, 14, -1)[:, np.newaxis, np.newaxis, np.newaxis]
    interpolatorType = "Linear"   # "Spline", "Linear", "Cubic"

    for nue in range(numUEsPerSlot):
        print(f"UE-Index: {ueIndices[nue]} | Slot-Index: {ns}")
        Hfest[:, nue] = chEST(chGrid, srsObject[nue], interpolatorType)[:, 0, 0, 0].reshape(nBSs, Nt, 14, -1)

    Hest = Hfest.sum(-2) / 14
    print(f"***********  [{ns}]-Kanal geschätzt *********** ")

    # --- ToA-Schätzung (ESPRIT) ---
    toaEstimation = ToAEstimation("ESPRIT", Hest[0, 0].T.shape)
    Lpath = 2

    for nbs in range(nBSs):
        for nue in range(numUEsPerSlot):
            print(f"(nbs, nue): ({nbs}, {ueIndices[nue]})")
            delayEstimates = np.sort(
                toaEstimation(Hest[nbs, nue].T, Lpath, subCarrierSpacing=scs)
            )
            delayEstimates = delayEstimates[delayEstimates > 0]
            K = Lpath
            while (delayEstimates.size == 0) or (delayEstimates[0] <= 0 and K < 12):
                K += 1
                delayEstimates = np.sort(
                    toaEstimation(Hest[nbs, nue].T, numberOfPath=K, subCarrierSpacing=scs)
                )
                delayEstimates = delayEstimates[delayEstimates > 0]

            if delayEstimates.size == 0:
                ToAe[nbs, ueIndices[nue]] = 1e-9
            else:
                ToAe[nbs, ueIndices[nue]] = delayEstimates[0]

    print(f"***********  [{ns}]-ToA geschätzt *********** ")


# =============================================================================
#  Positionsschätzung (TDoA – Least Squares)
# =============================================================================

rxPosition = simLayoutObj.UELocations
txPosition = simLayoutObj.BSLocations

k     = 4   # k beste Messungen auswählen
error = np.abs(ToAe - channel.delays[0, 0, ..., 0]) / channel.delays[0, 0, ..., 0]

positionEstimate   = LeastSquareTDoA()
rxPositionEstimate = np.zeros((nUEs, 2, 3))
rxStdEstimate      = np.zeros((nUEs))
kBestIndices       = np.zeros((nUEs, k), dtype=np.int8)

for nue in range(nUEs):
    bsIndices             = np.argmin(error[..., nue].reshape(-1, 3), axis=-1)
    siteIndices           = np.argsort(np.min(error[..., nue].reshape(-1, 3), -1))[0:k]
    kBestIndices[nue]     = siteIndices * nSectorsPerSite + bsIndices[siteIndices]
    toa                   = ToAe[kBestIndices[nue], nue]
    tdoa                  = toa[1:] - toa[0]
    rxPositionEstimate[nue], rxStdEstimate[nue] = positionEstimate(
        txPosition[kBestIndices[nue]], tdoa=tdoa
    )


# =============================================================================
#  Visualisierung 1: Topologie mit Entfernungsringen
# =============================================================================

rangeEst_2D = np.sqrt(
    np.abs((ToAe * 3e8) ** 2 - (rxPosition[:, 2].reshape(1, -1) - txPosition[:, 2].reshape(-1, 1)) ** 2)
)

fig, ax = simLayoutObj.display2DTopology(isEqualAspectRatio=True)

colors = ["k", "m", "r", "b", "g", "y", "crimson"]
linestyle_tuple = [
    'solid', 'dotted', 'dashed', 'dashdot',
    (0, (5, 10)),
    (0, (1, 10)),
    (5, (10, 3)),
    (0, (5, 1)),
    (0, (3, 10, 1, 10)),
    (0, (3, 5, 1, 5)),
    (0, (3, 1, 1, 1)),
    (0, (3, 5, 1, 5, 1, 5)),
    (0, (3, 10, 1, 10, 1, 10)),
    (0, (3, 1, 1, 1, 1, 1))
]

for nbs in range(k):
    for nue in range(nUEs):
        circle1 = plt.Circle(
            (txPosition[kBestIndices[nue, nbs], 0], txPosition[kBestIndices[nue, nbs], 1]),
            rangeEst_2D[kBestIndices[nue, nbs], nue],
            color=colors[nue % 7], lw=0.35, ls=linestyle_tuple[nue % 7],
            fill=False, zorder=0
        )
        ax.add_artist(circle1)

ax.scatter(txPosition[:, 0], txPosition[:, 1],
           marker="P", color="b", edgecolors='white', s=125, label="Tx-Positionen", zorder=3)
ax.scatter(rxPositionEstimate[:, 0, 0], rxPositionEstimate[:, 0, 1],
           marker="o", color="g", s=75, label="Geschätzte Rx-Positionen", zorder=1)
ax.scatter(rxPosition[:, 0], rxPosition[:, 1],
           marker=".", color="r", edgecolors='white', s=100, label="Wahre Rx-Positionen", zorder=2)

ax.legend()
ax.set_xlabel("x-Koordinaten (m)")
ax.set_ylabel("y-Koordinaten (m)")
ax.set_title("Tx-Positionen und Schätzgenauigkeit (Wahre vs. Geschätzte UE-Positionen)")
ax.set_xlim([-200, 200])
ax.set_ylim([-200, 200])
ax.grid(True)


# =============================================================================
#  Visualisierung 2: CDF des Positionierungsfehlers
# =============================================================================

nbins   = nUEs
xlimit  = 5
ylimit  = 1

posError3D = np.linalg.norm(rxPositionEstimate[:, 0] - rxPosition, axis=1)
posError3D = np.where(np.isnan(posError3D), 0, posError3D)
posError2D = np.linalg.norm(rxPositionEstimate[:, 0, 0:2] - rxPosition[:, 0:2], axis=1)

fig, ax = plt.subplots()

# 2D-Fehler
count, bins_count = np.histogram(posError2D, bins=nbins, range=[0, xlimit])
cdf = np.cumsum(count / nUEs)
ax.plot(bins_count[1:], cdf, label="2D-Positionierungsfehler")

# 3D-Fehler
count, bins_count = np.histogram(posError3D, bins=nbins, range=[0, xlimit])
cdf = np.cumsum(count / nUEs)
ax.plot(bins_count[1:], cdf, label="3D-Positionierungsfehler")

ax.set_xticks(np.linspace(0, xlimit, 11))
ax.set_xticks(np.linspace(0, xlimit, 21), minor=True)
ax.set_yticks(np.linspace(0, ylimit, 11))
ax.set_yticks(np.linspace(0, ylimit, 21), minor=True)
ax.set_xlabel("Positionierungsfehler (m)")
ax.set_ylabel("CDF des Positionierungsfehlers")
ax.set_title("CDF des Positionierungsfehlers")

ax.axhline(y=0.5,    lw=2, alpha=1, linestyle=':', color="crimson",     label="50%-Linie")
ax.axhline(y=2/3,    lw=2, alpha=1, linestyle=':', color="magenta",     label="66.6%-Linie")
ax.axhline(y=0.9,    lw=2, alpha=1, linestyle=':', color="royalblue",   label="90%-Linie")
ax.axvline(x=0.4,    lw=2, alpha=1, linestyle='-', color="midnightblue",label="50 cm Genauigkeitslinie")

ax.grid(which='minor', alpha=0.25, linestyle='--')
ax.grid(which='major', alpha=1)
ax.set_xlim([0, xlimit])
ax.set_ylim([0, ylimit])
ax.legend()
plt.show()


# =============================================================================
#  Optional: Ergebnisse speichern
# =============================================================================

# idx  = 0
# flag = True
# while flag:
#     filename = f"Databases/ULTDoA-[{idx}].npz"
#     if os.path.exists(filename):
#         idx += 1
#     else:
#         np.savez(filename,
#                  posError3D         = posError3D,
#                  posError2D         = posError2D,
#                  rxPositionEstimate = rxPositionEstimate,
#                  rxPosition         = rxPosition,
#                  ToAe               = ToAe,
#                  txPosition         = txPosition,
#                  propTerrain        = propTerrain,
#                  carrierFrequency   = carrierFrequency,
#                  scs                = scs,
#                  Nfft               = Nfft,
#                  nBSs               = nBSs,
#                  nUEs               = nUEs,
#                  numRBs             = numRBs,
#                  bsArrayStructure   = np.array([1, 1, 2, 2, 1]),
#                  ueArrayStructure   = np.array([1, 1, 8, 4, 1]))
#         flag = False
