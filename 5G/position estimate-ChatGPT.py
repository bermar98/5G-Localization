import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Helper Functions (vollständig integriert)
# ============================================================

def getgNBPositions(numgNBs):
    gNBPos = []
    for gNBIdx in range(numgNBs):
        phi = (gNBIdx * 2 * np.pi / numgNBs +
               np.random.rand() * 2 * np.pi / (2 * numgNBs) -
               2 * np.pi / (2 * numgNBs))

        r = np.random.randint(0, 1000) + 4000 + (gNBIdx * 5000 / numgNBs)

        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = 25

        gNBPos.append(np.array([x, y, z]))
    return gNBPos


def validateCarriers(carrier):
    cellIDs = [c.NCellID for c in carrier]
    if len(set(cellIDs)) != len(cellIDs):
        raise ValueError("Duplicate NCellID detected")

    scs = [c.SubcarrierSpacing for c in carrier]
    if len(set(scs)) != 1:
        raise ValueError("Subcarrier spacing mismatch")


def validateNumLayers(pdsch):
    if any(p.NumLayers != 1 for p in pdsch):
        raise ValueError("Only NumLayers = 1 supported")


def getRSTDValues(toa, sr):
    toa = np.array(toa)
    rstd = np.zeros((len(toa), len(toa)))

    for i in range(len(toa)):
        for j in range(len(toa)):
            rstd[i, j] = (toa[i] - toa[j]) / sr

    return rstd


def getRSTDCurve(gNB1, gNB2, rstd):
    delta = gNB1 - gNB2

    phi = np.arctan2(delta[1], delta[0])
    r = np.linalg.norm(delta)

    rd = (r + rstd) / 2

    a = (r / 2) - rd
    c = r / 2
    b = np.sqrt(np.abs(c**2 - a**2))

    hk = (gNB1 + gNB2) / 2

    mu = np.linspace(-2, 2, 1000)

    x = a * np.cosh(mu) * np.cos(phi) - b * np.sinh(mu) * np.sin(phi) + hk[0]
    y = a * np.cosh(mu) * np.sin(phi) + b * np.sinh(mu) * np.cos(phi) + hk[1]

    return x, y


def getEstimatedUEPosition(xCell, yCell):
    x_points = []
    y_points = []

    for i in range(len(xCell)):
        for j in range(i + 1, len(xCell)):
            x1, y1 = xCell[i], yCell[i]
            x2, y2 = xCell[j], yCell[j]

            idx = np.argmin(np.abs(x1 - x2))

            x_points.append((x1[idx] + x2[idx]) / 2)
            y_points.append((y1[idx] + y2[idx]) / 2)

    return np.array([np.mean(x_points), np.mean(y_points), 0])


def plotgNBAndUEPositions(gNBPos, UEPos, gNBNums):
    plt.figure()

    for k, idx in enumerate(gNBNums):
        p = gNBPos[idx]
        plt.scatter(p[0], p[1], marker="^", label=f"gNB{k+1}")

    plt.scatter(UEPos[0], UEPos[1], c="black", label="UE")

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.legend()
    plt.show()


# ============================================================
# Simulation Parameters
# ============================================================

nFrames = 1
fc = 3e9

UEPos = np.array([500, -20, 2])

numgNBs = 5
np.random.seed(1)

gNBPos = getgNBPositions(numgNBs)

plotgNBAndUEPositions(gNBPos, UEPos, np.arange(numgNBs))

# ============================================================
# Carrier Configuration
# ============================================================

cellIds = np.random.permutation(1008)[:numgNBs]

carrier = [nrCarrierConfig() for _ in range(numgNBs)]

for i in range(numgNBs):
    carrier[i].NCellID = int(cellIds[i])

validateCarriers(carrier)

# ============================================================
# PRS Configuration
# ============================================================

prsSlotOffsets = np.arange(0, 2 * numgNBs, 2)
prsIDs = np.random.permutation(4096)[:numgNBs]

prs = [nrPRSConfig() for _ in range(numgNBs)]

for i in range(numgNBs):
    prs[i].PRSResourceSetPeriod = [10, 0]
    prs[i].PRSResourceOffset = int(prsSlotOffsets[i])
    prs[i].NPRSID = int(prsIDs[i])

    prs[i].NumRB = 52
    prs[i].CombSize = 12
    prs[i].NumPRSSymbols = 12

# ============================================================
# PDSCH
# ============================================================

pdsch = [nrPDSCHConfig() for _ in range(numgNBs)]

for i in range(numgNBs):
    pdsch[i].PRBSet = np.arange(52)
    pdsch[i].SymbolAllocation = [0, 14]
    pdsch[i].NumLayers = 1

validateNumLayers(pdsch)

# ============================================================
# Resource Generation
# ============================================================

totSlots = nFrames * carrier[0].SlotsPerFrame

prsGrid = [[] for _ in range(numgNBs)]
dataGrid = [[] for _ in range(numgNBs)]

for slot in range(totSlots):

    for i in range(numgNBs):
        carrier[i].NSlot = slot
        slotGrid = nrResourceGrid(carrier[i], 1)

        prsSym = nrPRS(carrier[i], prs[i])
        prsInd = nrPRSIndices(carrier[i], prs[i])

        slotGrid[prsInd] = prsSym
        prsGrid[i].append(slotGrid)

    for i in range(numgNBs):
        dataSlot = nrResourceGrid(carrier[i], 1)

        if all(len(nrPRSIndices(carrier[j], prs[j])) == 0 for j in range(numgNBs)):
            ind, info = nrPDSCHIndices(carrier[i], pdsch[i])
            bits = np.random.randint(0, 2, info.G)

            sym = nrPDSCH(carrier[i], pdsch[i], bits)

            dmrsInd = nrPDSCHDMRSIndices(carrier[i], pdsch[i])
            dmrsSym = nrPDSCHDMRS(carrier[i], pdsch[i])

            dataSlot[ind] = sym
            dataSlot[dmrsInd] = dmrsSym

        dataGrid[i].append(dataSlot)

# ============================================================
# OFDM Modulation
# ============================================================

txWaveform = []

for i in range(numgNBs):
    carrier[i].NSlot = 0
    grid = np.array(prsGrid[i]) + np.array(dataGrid[i])
    txWaveform.append(nrOFDMModulate(carrier[i], grid))

ofdmInfo = nrOFDMInfo(carrier[0])

# ============================================================
# Channel Simulation
# ============================================================

c = 3e8
sampleDelay = np.zeros(numgNBs, dtype=int)

rxWaveform = None

for i in range(numgNBs):

    dist = np.linalg.norm(gNBPos[i] - UEPos)
    delay = dist / c

    sampleDelay[i] = int(round(delay * ofdmInfo.SampleRate))

    PLdB = nrPathLoss(nrPathLossConfig(), fc, True, gNBPos[i], UEPos)
    PL = 10 ** (PLdB / 10)

    sig = np.concatenate([
        np.zeros(sampleDelay[i]),
        txWaveform[i],
        np.zeros(max(sampleDelay) - sampleDelay[i])
    ]) / np.sqrt(PL)

    rxWaveform = sig if rxWaveform is None else rxWaveform + sig

# ============================================================
# TOA estimation
# ============================================================

corr = []
delayEst = []
maxCorr = []

for i in range(numgNBs):
    _, mag = nrTimingEstimate(carrier[i], rxWaveform, prsGrid[i])

    c = mag[:1000]

    corr.append(c)
    maxCorr.append(np.max(c))
    delayEst.append(np.argmax(c))

detected = np.argsort(maxCorr)[::-1]
detected = detected[:min(3, numgNBs)]

# ============================================================
# RSTD + Hyperbolas
# ============================================================

rstdVals = getRSTDValues(delayEst, ofdmInfo.SampleRate)

curveX = []
curveY = []
gNBNums = []

ref = detected[0]

for i in detected[1:]:

    rstd = rstdVals[i, ref] * c

    x, y = getRSTDCurve(gNBPos[ref], gNBPos[i], rstd)

    curveX.append(x)
    curveY.append(y)
    gNBNums.append([ref, i])

# ============================================================
# Position estimation
# ============================================================

estPos = getEstimatedUEPosition(curveX, curveY)

print("Estimated UE Position:", estPos)
print("Error:", np.linalg.norm(UEPos - estPos))

# ============================================================
# Plot result
# ============================================================

plotgNBAndUEPositions(gNBPos, UEPos, detected)