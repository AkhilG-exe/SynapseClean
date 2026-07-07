# SynapseClean 🧠

An automated, standalone neuro-signal conditioning pipeline that simulates, analyzes, and purifies transient high-frequency physiological artifacts out of synthetic clinical-rate multi-band EEG telemetry arrays.

## 🚀 Overview

Physiological artifacts (such as electromyographic/muscle interference) heavily distort raw EEG data, rendering raw telemetry challenging for downstream ML applications. **SynapseClean** provides a production-grade demonstration of digital signal conditioning. It synthesizes complex multi-band neural waveforms, introduces high-amplitude high-frequency noise transients, and implements an algorithmic zero-phase attenuation framework to cleanly recover the pristine underlying signal.

## ⚡ Key Features

* **Synthetic Waveform Synthesis:** Compiles a realistic 5-second neural array blending Alpha ($10\text{ Hz}$, $30\text{ }\mu\text{V}$) and Delta ($4\text{ Hz}$, $60\text{ }\mu\text{V}$) bands sampled at a clinical frequency of $250\text{ Hz}$.
* **Myogenic Artifact Injection:** Simulates acute, high-amplitude electromyographic (EMG) noise ($60\text{ Hz}$, $100\text{ }\mu\text{V}$) targeting the center of the timeline window to challenge extraction filters.
* **Zero-Phase Digital Conditioning:** Deploys a 4th-order digital Butterworth bandpass filter ($1.0\text{--}30.0\text{ Hz}$) utilizing `scipy.signal.filtfilt` to wipe away artifacts without introducing phase or timing shifts.
* **Automated Dependency Engine:** Built-in verification loop programmatically catches and installs missing core scientific libraries via subprocess channels for an instant 1-click execution experience.

## 📊 Analytical Performance Visualization

Upon running execution, the script automatically exports a high-fidelity comparative visual analytics array saved as `eeg_filtering_results.png`:

* **Top Subplot:** Raw input data stream showcasing severe voltage distortion within the designated muscle contraction envelope.
* **Bottom Subplot:** Pure signal extraction isolating recovered rhythmic alpha/delta components returned precisely to true phase baselines.

## 📦 Installation & Quickstart

No manual setup required. Simply clone the repository and execute the script directly—the engine handles package allocation out-of-the-box:

```bash
# Clone the repository
git clone [https://github.com/AkhilG-exe/SynapseClean.git](https://github.com/AkhilG.exe/SynapseClean.git)
cd SynapseClean

# Run the pipeline
python eeg_cleaner.py
