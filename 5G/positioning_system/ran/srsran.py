# =============================================================================
#  ran/srsran.py – Phase 2: Echte IQ-Daten von srsRAN via ZMQ
#
#  Diese Klasse implementiert dieselbe DataSource-Schnittstelle wie
#  SimulatedSource – der gesamte Signalverarbeitungs-Code bleibt identisch.
#
#  Voraussetzungen für Phase 2:
#    - srsRAN Project läuft mit ZMQ-Backend (statt USRP)
#    - Open5GS als 5G Core Network
#    - pip install pyzmq
#
#  Architektur:
#    srsRAN gNB 1 (ZMQ Port 2000) ─┐
#    srsRAN gNB 2 (ZMQ Port 2002) ──┼─► SrsRanSource.get_channel_matrix()
#    srsRAN gNB 3 (ZMQ Port 2004) ─┘
#
#  Datenfluss pro gNB:
#    IQ-Samples (ZMQ) → OFDM-Demod → Kanalschätzung → H[nbs, nue, Nsc]
# =============================================================================

import numpy as np
import config
from ran.base import DataSource


class SrsRanSource(DataSource):
    """
    Liest IQ-Daten von srsRAN gNBs via ZMQ und berechnet H(f).

    Jeder gNB läuft in srsRAN mit ZMQ als RF-Frontend:
        ru_sdr:
          device_driver: zmq
          tx_port: tcp://*:200X
          rx_port: tcp://localhost:200Y

    Diese Klasse verbindet sich zu den ZMQ-Ports, empfängt die
    IQ-Samples, führt OFDM-Demodulation und Kanalschätzung durch
    und gibt H[nbs, nue, Nsc] zurück – identisch zu SimulatedSource.
    """

    def __init__(self,
                 bs_positions: np.ndarray,
                 ue_positions: np.ndarray,
                 zmq_ports: list = None):
        """
        Parameters
        ----------
        bs_positions : np.ndarray, shape (n_bs, 3)
            Bekannte BS-Positionen im Gebäude [m].
            Werden beim Aufbau des Campusnetzes einmalig vermessen.

        ue_positions : np.ndarray, shape (n_ue, 3) oder None
            Referenz-UE-Positionen (nur für Evaluierung / Kalibrierung).
            Im Produktivbetrieb: None (werden ja geschätzt).

        zmq_ports : list of int
            ZMQ-Ports der srsRAN gNBs, z.B. [2000, 2002, 2004].
            Muss len(bs_positions) Einträge haben.
        """
        self._txPos = np.array(bs_positions)
        self._rxPos = np.array(ue_positions) if ue_positions is not None else None
        self._ports = zmq_ports or config.ZMQ_GNB_PORTS[:len(bs_positions)]
        self._sockets = []
        self._H = None
        self._delays = None

        assert len(self._ports) == self.n_bs, (
            f"Anzahl ZMQ-Ports ({len(self._ports)}) muss "
            f"Anzahl BSs ({self.n_bs}) entsprechen."
        )

        self._connect_zmq()

    # -------------------------------------------------------------------------
    #  DataSource Interface
    # -------------------------------------------------------------------------

    def get_channel_matrix(self) -> np.ndarray:
        """
        Empfängt IQ-Samples von allen gNBs und berechnet H[nbs, nue, Nsc].

        Ablauf pro gNB:
        1. IQ-Samples empfangen (ZMQ)
        2. OFDM-Demodulation → Ressourcengitter
        3. SRS-Extraktion aus Gitter
        4. LS-Kanalschätzung → H_pilot
        5. Interpolation → H[Nsc]
        """
        # Lazy Import – nur wenn tatsächlich genutzt
        from signal_processing.ofdm import ofdm_demodulate
        from signal_processing.srs_generator import generate_srs_grid
        from signal_processing.channel_estimation import estimate_channel_ls

        n_ue_ = self.n_ue
        H = np.zeros((self.n_bs, n_ue_, config.NSC), dtype=np.complex64)

        for nbs, sock in enumerate(self._sockets):
            # IQ-Samples empfangen
            raw = sock.recv()
            iq_samples = np.frombuffer(raw, dtype=np.complex64)

            for nue in range(n_ue_):
                # OFDM-Demodulation
                rx_grid = ofdm_demodulate(iq_samples, config.NFFT,
                                          config.NSC, nSymbols=14)

                # SRS-Indizes und gesendete Symbole für dieses UE
                _, srs_idx, tx_sym = generate_srs_grid(
                    config.NSC, nue,
                    config.SRS_TRANSMISSION_COMB,
                    config.SRS_SYMBOL_START,
                    config.SRS_CSRS,
                    config.SRS_BSRS,
                    config.SRS_CYCLIC_SHIFT
                )

                # LS-Kanalschätzung
                H[nbs, nue] = estimate_channel_ls(rx_grid, srs_idx, tx_sym)

        self._H = H
        return H.copy()

    def get_true_delays(self) -> np.ndarray:
        """
        Berechnet geometrische Delays aus bekannten Positionen.
        Nur für Evaluierung verfügbar wenn UE-Referenzpositionen bekannt sind.
        """
        if self._rxPos is None:
            raise RuntimeError(
                "Wahre UE-Positionen nicht bekannt. "
                "Nur im Evaluierungsmodus verfügbar."
            )
        delays = np.zeros((self.n_bs, self.n_ue))
        for nbs in range(self.n_bs):
            for nue in range(self.n_ue):
                d3d = np.linalg.norm(
                    self._rxPos[nue] - self._txPos[nbs]
                )
                delays[nbs, nue] = d3d / config.C
        return delays

    def get_bs_positions(self) -> np.ndarray:
        return self._txPos.copy()

    def get_ue_positions(self) -> np.ndarray:
        if self._rxPos is None:
            raise RuntimeError("UE-Positionen nicht bekannt (Produktivmodus).")
        return self._rxPos.copy()

    @property
    def n_bs(self) -> int:
        return len(self._txPos)

    @property
    def n_ue(self) -> int:
        return len(self._rxPos) if self._rxPos is not None else 1

    # -------------------------------------------------------------------------
    #  ZMQ-Verbindung
    # -------------------------------------------------------------------------

    def _connect_zmq(self):
        """Verbindet zu allen srsRAN gNB ZMQ-Ports."""
        try:
            import zmq
        except ImportError:
            raise ImportError(
                "pyzmq ist nicht installiert.\n"
                "Installation: pip install pyzmq\n"
                "Nur für Phase 2 (echtes srsRAN) benötigt."
            )

        ctx = zmq.Context()
        for port in self._ports:
            sock = ctx.socket(zmq.SUB)
            sock.connect(f"tcp://localhost:{port}")
            sock.setsockopt(zmq.SUBSCRIBE, b"")
            sock.setsockopt(zmq.RCVTIMEO, config.ZMQ_TIMEOUT_MS)
            self._sockets.append(sock)
            print(f"[SrsRanSource] Verbunden mit ZMQ Port {port}")

    def close(self):
        """Schließt alle ZMQ-Verbindungen."""
        for sock in self._sockets:
            sock.close()
        print("[SrsRanSource] ZMQ-Verbindungen geschlossen.")

    def __del__(self):
        if self._sockets:
            self.close()

    # -------------------------------------------------------------------------
    #  Konfigurationshilfe für Phase 2
    # -------------------------------------------------------------------------

    @staticmethod
    def print_srsran_config(n_bs: int, base_port: int = 2000):
        """
        Gibt die srsRAN gnb_zmq.yaml Konfiguration für alle gNBs aus.
        Hilfreich beim Aufbau des Campusnetzes.
        """
        print("\n" + "="*60)
        print("srsRAN ZMQ Konfiguration (gnb_zmq.yaml)")
        print("="*60)
        for i in range(n_bs):
            tx_port = base_port + i*2
            rx_port = base_port + i*2 + 1
            print(f"\n# gNB {i+1}")
            print(f"ru_sdr:")
            print(f"  device_driver: zmq")
            print(f"  tx_port: tcp://*:{tx_port}")
            print(f"  rx_port: tcp://localhost:{rx_port}")
            print(f"  srate: 11.52  # MHz für 10 MHz Bandbreite")
        print("="*60 + "\n")
