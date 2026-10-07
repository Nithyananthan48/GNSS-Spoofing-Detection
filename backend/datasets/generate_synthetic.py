"""
GNSS Spoofing Dataset Synthetic Generator
==========================================
Generates a realistic synthetic GNSS dataset for development and testing.

This dataset simulates:
- Normal GNSS observations with realistic noise
- Spoofed GNSS observations with controlled anomalies:
  * Sudden pseudorange jumps
  * C/N0 drops
  * Carrier phase discontinuities
  * Correlated position errors
  * Abnormal Doppler variations

IMPORTANT: This is a SYNTHETIC dataset for development/testing only.
Replace with a real GNSS spoofing dataset for production use.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent


def generate_gnss_dataset(
    n_normal: int = 3000,
    n_spoofed: int = 2000,
    output_file: str = None,
) -> pd.DataFrame:
    """
    Generate synthetic GNSS time-series data with normal and spoofed observations.

    Features:
        pseudorange     - satellite-to-receiver pseudorange (m)
        carrier_phase   - carrier phase measurement (cycles)
        cn0             - carrier-to-noise density ratio (dB-Hz)
        doppler         - Doppler frequency shift (Hz)
        elevation       - satellite elevation angle (degrees)
        azimuth         - satellite azimuth angle (degrees)
        latitude        - receiver latitude (degrees)
        longitude       - receiver longitude (degrees)
        altitude        - receiver altitude (m)
        velocity        - receiver velocity (m/s)
        timestamp       - Unix timestamp (seconds)
        label           - 0 = Normal, 1 = Spoofed
    """

    def generate_normal(n: int, start_time: float = 0.0) -> pd.DataFrame:
        """Generate normal (non-spoofed) GNSS observations."""
        t = np.arange(n) * 1.0  # 1 second intervals
        timestamps = start_time + t

        # Pseudorange: ~20,000 km with small noise
        pseudorange = 20_000_000 + np.cumsum(np.random.normal(0, 0.5, n))

        # Carrier phase: proportional to pseudorange with small noise
        carrier_phase = pseudorange / 0.1903 + np.random.normal(0, 0.01, n)

        # C/N0: normally 35-45 dB-Hz
        cn0 = 40 + np.random.normal(0, 1.5, n)
        cn0 = np.clip(cn0, 25, 55)

        # Doppler: small variations around 0 Hz
        doppler = np.random.normal(0, 5, n)

        # Elevation: 10-85 degrees, slowly varying
        elevation = 45 + 20 * np.sin(2 * np.pi * t / 600) + np.random.normal(0, 1, n)
        elevation = np.clip(elevation, 10, 85)

        # Azimuth: 0-360, slowly changing
        azimuth = (180 + 60 * np.sin(2 * np.pi * t / 900) + np.random.normal(0, 2, n)) % 360

        # Position: nearly static with small GPS noise (~0.5m)
        base_lat, base_lon, base_alt = 28.6139, 77.2090, 216.0  # New Delhi
        latitude = base_lat + np.cumsum(np.random.normal(0, 0.000002, n))
        longitude = base_lon + np.cumsum(np.random.normal(0, 0.000002, n))
        altitude = base_alt + np.random.normal(0, 1.5, n)

        # Velocity: near-zero (static receiver)
        velocity = np.abs(np.random.normal(0.0, 0.3, n))

        return pd.DataFrame({
            "timestamp": timestamps,
            "pseudorange": pseudorange,
            "carrier_phase": carrier_phase,
            "cn0": cn0,
            "doppler": doppler,
            "elevation": elevation,
            "azimuth": azimuth,
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude,
            "velocity": velocity,
            "label": 0,
        })

    def generate_spoofed(n: int, start_time: float = 5000.0) -> pd.DataFrame:
        """Generate spoofed GNSS observations with controlled anomalies."""
        t = np.arange(n) * 1.0
        timestamps = start_time + t

        # Pseudorange: sudden jumps simulating spoofing signal
        pseudorange_base = 20_000_000 + np.cumsum(np.random.normal(0, 0.5, n))
        # Add spoofing jumps at random intervals
        jump_indices = np.random.choice(n, size=max(1, n // 50), replace=False)
        jump_amounts = np.random.uniform(50, 500, len(jump_indices))  # 50-500m jumps
        for idx, jump in zip(sorted(jump_indices), jump_amounts):
            pseudorange_base[idx:] += jump
        pseudorange = pseudorange_base

        # Carrier phase: discontinuities (cycle slips)
        carrier_phase = pseudorange / 0.1903 + np.random.normal(0, 0.05, n)
        slip_indices = np.random.choice(n, size=max(1, n // 100), replace=False)
        for idx in sorted(slip_indices):
            carrier_phase[idx:] += np.random.uniform(100, 1000)

        # C/N0: drops during spoofing (spoofed signals often have lower C/N0)
        cn0 = 30 + np.random.normal(0, 3, n)  # lower mean, more variance
        # Periodic severe drops
        drop_indices = np.random.choice(n, size=max(1, n // 30), replace=False)
        for idx in sorted(drop_indices):
            length = min(np.random.randint(5, 20), n - idx)
            cn0[idx:idx + length] -= np.random.uniform(8, 15)
        cn0 = np.clip(cn0, 15, 55)

        # Doppler: abnormal variations (spoofed signal has different Doppler)
        doppler = np.random.normal(0, 15, n)  # higher variance than normal
        doppler_jumps = np.random.choice(n, size=max(1, n // 40), replace=False)
        for idx in sorted(doppler_jumps):
            doppler[idx] += np.random.uniform(-50, 50)

        # Elevation: may fluctuate more under spoofing
        elevation = 45 + 20 * np.sin(2 * np.pi * t / 600) + np.random.normal(0, 3, n)
        elevation = np.clip(elevation, 5, 90)

        # Azimuth: more erratic under spoofing
        azimuth = (180 + 60 * np.sin(2 * np.pi * t / 900) + np.random.normal(0, 8, n)) % 360

        # Position: large jumps and drift from true position
        base_lat, base_lon, base_alt = 28.6139, 77.2090, 216.0
        lat_drift = np.cumsum(np.random.normal(0, 0.00005, n))  # ~5m/s drift
        lon_drift = np.cumsum(np.random.normal(0, 0.00005, n))
        latitude = base_lat + lat_drift + np.random.normal(0, 0.000005, n)
        longitude = base_lon + lon_drift + np.random.normal(0, 0.000005, n)
        altitude = base_alt + np.cumsum(np.random.normal(0, 2, n)) + np.random.normal(0, 5, n)

        # Velocity: higher and more erratic (spoofing causes phantom movement)
        velocity = np.abs(np.random.normal(2.0, 1.5, n))

        return pd.DataFrame({
            "timestamp": timestamps,
            "pseudorange": pseudorange,
            "carrier_phase": carrier_phase,
            "cn0": cn0,
            "doppler": doppler,
            "elevation": elevation,
            "azimuth": azimuth,
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude,
            "velocity": velocity,
            "label": 1,
        })

    print(f"Generating {n_normal} normal observations...")
    df_normal = generate_normal(n_normal, start_time=0.0)

    print(f"Generating {n_spoofed} spoofed observations...")
    df_spoofed = generate_spoofed(n_spoofed, start_time=float(n_normal))

    # Combine and shuffle
    df = pd.concat([df_normal, df_spoofed], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nDataset Summary:")
    print(f"  Total observations: {len(df)}")
    print(f"  Normal   (label=0): {(df['label'] == 0).sum()}")
    print(f"  Spoofed  (label=1): {(df['label'] == 1).sum()}")
    print(f"  Features: {[c for c in df.columns if c != 'label' and c != 'timestamp']}")

    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\nDataset saved to: {output_file}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic GNSS dataset")
    parser.add_argument("--normal", type=int, default=3000, help="Number of normal samples")
    parser.add_argument("--spoofed", type=int, default=2000, help="Number of spoofed samples")
    parser.add_argument(
        "--output",
        type=str,
        default=str(BASE_DIR / "synthetic_gnss_dataset.csv"),
        help="Output CSV file path",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GNSS Spoofing Synthetic Dataset Generator")
    print("NOTE: This is a DEVELOPMENT/TESTING dataset only.")
    print("=" * 60)
    generate_gnss_dataset(
        n_normal=args.normal,
        n_spoofed=args.spoofed,
        output_file=args.output,
    )
