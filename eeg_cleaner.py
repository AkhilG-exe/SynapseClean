#!/usr/bin/env python3

import sys
import subprocess
import importlib


def install_package(package_name):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False


def ensure_dependencies():
    """Ensure required packages are installed."""
    required_packages = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib'
    }

    for import_name, pip_name in required_packages.items():
        try:
            importlib.import_module(import_name)
            print(f"[OK] {import_name} already installed")
        except ImportError:
            print(f"Installing {pip_name}...")
            if install_package(pip_name):
                print(f"[OK] {pip_name} installed successfully")
            else:
                print(f"[FAIL] Failed to install {pip_name}")
                sys.exit(1)


# Ensure dependencies before importing
ensure_dependencies()

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def generate_synthetic_eeg(duration=5.0, fs=250):
    """
    Generate a synthetic EEG signal with Alpha (10 Hz) and Delta (4 Hz) waves.

    Parameters:
    - duration: Signal duration in seconds
    - fs: Sampling frequency in Hz

    Returns:
    - t: Time vector
    - eeg: Synthetic EEG signal
    """
    # Time vector
    t = np.arange(0, duration, 1/fs)

    # Generate Alpha wave (10 Hz) - typical amplitude ~20-50 µV
    alpha = 30 * np.sin(2 * np.pi * 10 * t)

    # Generate Delta wave (4 Hz) - typical amplitude ~50-100 µV
    delta = 60 * np.sin(2 * np.pi * 4 * t)

    # Combine waves
    eeg = alpha + delta

    # Add some background noise (low amplitude)
    np.random.seed(42)  # For reproducibility
    eeg += 5 * np.random.randn(len(t))

    return t, eeg


def inject_muscle_artifact(eeg, t, fs=250, artifact_freq=60, artifact_duration=0.5, artifact_amplitude=100):
    """
    Inject a sudden high-frequency muscle artifact (60 Hz) in the middle of the signal.

    Parameters:
    - eeg: Original EEG signal
    - t: Time vector
    - fs: Sampling frequency
    - artifact_freq: Frequency of the artifact in Hz
    - artifact_duration: Duration of artifact in seconds
    - artifact_amplitude: Amplitude of the artifact

    Returns:
    - eeg_with_artifact: EEG signal with artifact injected
    - artifact_mask: Boolean mask indicating artifact location
    """
    eeg_with_artifact = eeg.copy()
    artifact_mask = np.zeros_like(eeg, dtype=bool)

    # Find middle of signal
    mid_idx = len(eeg) // 2
    artifact_samples = int(artifact_duration * fs)
    start_idx = mid_idx - artifact_samples // 2
    end_idx = start_idx + artifact_samples

    # Ensure indices are within bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(eeg), end_idx)

    # Generate artifact
    t_artifact = t[start_idx:end_idx]
    artifact = artifact_amplitude * np.sin(2 * np.pi * artifact_freq * t_artifact)

    # Inject artifact
    eeg_with_artifact[start_idx:end_idx] += artifact
    artifact_mask[start_idx:end_idx] = True

    return eeg_with_artifact, artifact_mask


def apply_bandpass_filter(data, fs, lowcut=1.0, highcut=30.0, order=4):
    """
    Apply a Butterworth bandpass filter to the signal.

    Parameters:
    - data: Input signal
    - fs: Sampling frequency
    - lowcut: Low cutoff frequency in Hz
    - highcut: High cutoff frequency in Hz
    - order: Filter order

    Returns:
    - filtered: Filtered signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design Butterworth bandpass filter
    b, a = signal.butter(order, [low, high], btype='band')

    # Apply filter (zero-phase filtering using filtfilt)
    filtered = signal.filtfilt(b, a, data)

    return filtered


def plot_results(t, raw_signal, filtered_signal, artifact_mask, fs, output_file="eeg_filtering_results.png"):
    """
    Plot Raw vs Filtered signals side-by-side.

    Parameters:
    - t: Time vector
    - raw_signal: Raw EEG signal with artifact
    - filtered_signal: Filtered EEG signal
    - artifact_mask: Boolean mask of artifact location
    - fs: Sampling frequency
    - output_file: Output filename
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Raw signal with artifact highlighted
    ax1 = axes[0]
    ax1.plot(t, raw_signal, 'b-', linewidth=0.8, label='Raw EEG (with 60 Hz artifact)')

    # Highlight artifact region
    artifact_indices = np.where(artifact_mask)[0]
    if len(artifact_indices) > 0:
        artifact_start = t[artifact_indices[0]]
        artifact_end = t[artifact_indices[-1]]
        ax1.axvspan(artifact_start, artifact_end, alpha=0.3, color='red',
                    label=f'60 Hz Muscle Artifact ({artifact_end - artifact_start:.2f}s)')

    ax1.set_ylabel('Amplitude (µV)', fontsize=12)
    ax1.set_title('Raw EEG Signal with Injected 60 Hz Muscle Artifact', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-250, 250)

    # Plot 2: Filtered signal
    ax2 = axes[1]
    ax2.plot(t, filtered_signal, 'g-', linewidth=0.8, label='Filtered EEG (1-30 Hz Butterworth)')

    # Also show original clean signal region for comparison (outside artifact)
    clean_mask = ~artifact_mask
    if np.any(clean_mask):
        ax2.plot(t[clean_mask], filtered_signal[clean_mask], 'g-', linewidth=0.8)

    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Amplitude (µV)', fontsize=12)
    ax2.set_title('Filtered EEG Signal (Butterworth Bandpass 1-30 Hz)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-120, 120)

    # Set x-axis limits
    ax2.set_xlim(0, t[-1])

    # Add overall title
    fig.suptitle(f'EEG Artifact Removal: Synthetic 5s EEG @ {fs} Hz\n'
                 f'Alpha (10 Hz) + Delta (4 Hz) with 60 Hz Muscle Artifact Removal',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[OK] Figure saved as: {output_file}")
    plt.close()


def main():
    """Main function to run the EEG cleaning pipeline."""
    print("=" * 60)
    print("EEG Signal Cleaner - Synthetic Signal with Artifact Removal")
    print("=" * 60)

    # Parameters
    duration = 5.0      # seconds
    fs = 250            # Hz

    print(f"\nGenerating synthetic EEG signal...")
    print(f"  Duration: {duration} seconds")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Components: Alpha (10 Hz) + Delta (4 Hz)")

    # Generate synthetic EEG
    t, clean_eeg = generate_synthetic_eeg(duration, fs)
    print(f"  Signal length: {len(t)} samples")

    # Inject muscle artifact
    print(f"\nInjecting 60 Hz muscle artifact in the middle...")
    raw_eeg, artifact_mask = inject_muscle_artifact(clean_eeg, t, fs)
    artifact_duration = np.sum(artifact_mask) / fs
    print(f"  Artifact duration: {artifact_duration:.2f} seconds")
    print(f"  Artifact frequency: 60 Hz")

    # Apply bandpass filter
    print(f"\nApplying Butterworth bandpass filter (1.0 - 30.0 Hz)...")
    filtered_eeg = apply_bandpass_filter(raw_eeg, fs, lowcut=1.0, highcut=30.0, order=4)
    print(f"  Filter order: 4")
    print(f"  Filter type: Zero-phase (filtfilt)")

    # Calculate SNR improvement
    artifact_region = raw_eeg[artifact_mask]
    filtered_artifact_region = filtered_eeg[artifact_mask]

    # Power in artifact region before and after
    raw_power = np.mean(artifact_region**2)
    filtered_power = np.mean(filtered_artifact_region**2)
    snr_improvement = 10 * np.log10(raw_power / filtered_power) if filtered_power > 0 else float('inf')

    print(f"\nArtifact suppression: {snr_improvement:.1f} dB reduction in artifact power")

    # Plot results
    print(f"\nGenerating visualization...")
    plot_results(t, raw_eeg, filtered_eeg, artifact_mask, fs)

    print("\n" + "=" * 60)
    print("EEG cleaning complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()