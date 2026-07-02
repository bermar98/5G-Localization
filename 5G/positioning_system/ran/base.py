# =============================================================================
#  ran/base.py – Abstrakte Basisklasse für alle Datenquellen
#
#  Jede Datenquelle (simuliert oder echt) implementiert dieses Interface.
#  Der gesamte Signalverarbeitungs- und Positioning-Code arbeitet
#  ausschließlich gegen dieses Interface – nie direkt gegen eine
#  konkrete Implementierung.
# =============================================================================

from abc import ABC, abstractmethod
import numpy as np


class DataSource(ABC):
    """
    Abstrakte Basisklasse für RAN-Datenquellen.

    Phase 1: SimulatedSource  → synthetischer TDL-Kanal
    Phase 2: SrsRanSource     → echte IQ-Daten von srsRAN via ZMQ

    Beide liefern dieselben Ausgaben – der Rest des Systems merkt
    keinen Unterschied.
    """

    @abstractmethod
    def get_channel_matrix(self) -> np.ndarray:
        """
        Gibt den komplexen Frequenzgang H zurück.

        Returns
        -------
        H : np.ndarray, shape (n_bs, n_ue, Nsc), dtype complex64
            H[nbs, nue, :] ist der Frequenzgang zwischen BS nbs und UE nue
            über alle Nsc Subcarrier.
        """
        raise NotImplementedError

    @abstractmethod
    def get_true_delays(self) -> np.ndarray:
        """
        Gibt die wahren geometrischen Laufzeiten zurück.
        Im echten System: aus BS-Koordinaten + UE-Startposition (nur für Eval).

        Returns
        -------
        delays : np.ndarray, shape (n_bs, n_ue), dtype float64
                 delays[nbs, nue] = d_3d / c  [s]
        """
        raise NotImplementedError

    @abstractmethod
    def get_bs_positions(self) -> np.ndarray:
        """
        Gibt die BS-Positionen zurück.

        Returns
        -------
        txPos : np.ndarray, shape (n_bs, 3), dtype float64
                Spalten: [x, y, z] in Metern
        """
        raise NotImplementedError

    @abstractmethod
    def get_ue_positions(self) -> np.ndarray:
        """
        Gibt die (wahren) UE-Positionen zurück.
        Im echten System: Referenzpositionen für Evaluierung.

        Returns
        -------
        rxPos : np.ndarray, shape (n_ue, 3), dtype float64
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def n_bs(self) -> int:
        """Anzahl Basisstationen."""
        raise NotImplementedError

    @property
    @abstractmethod
    def n_ue(self) -> int:
        """Anzahl UEs."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}"
                f"(n_bs={self.n_bs}, n_ue={self.n_ue})")
