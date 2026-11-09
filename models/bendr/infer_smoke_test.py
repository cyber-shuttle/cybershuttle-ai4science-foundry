import os
import torch
import collections
import collections.abc

# Compatibility patch for older libraries running on Python 3.10+
for abc_name in ['Iterable', 'Mapping', 'MutableMapping', 'Sequence', 'Callable', 'Set']:
    if not hasattr(collections, abc_name):
        setattr(collections, abc_name, getattr(collections.abc, abc_name))

# Compatibility patch for MOABB dataset naming changes
import moabb.datasets as mbd
if not hasattr(mbd, 'BNCI2014001') and hasattr(mbd, 'BNCI2014_001'):
    mbd.BNCI2014001 = mbd.BNCI2014_001

from dn3_ext import BENDRClassification

def test_bendr_inference():
    print("Initializing BENDR inference smoke test...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Portable path configuration relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, "checkpoints")
    
    encoder_checkpoint = os.path.join(checkpoint_dir, "encoder.pt")
    contextualizer_checkpoint = os.path.join(checkpoint_dir, "contextualizer.pt")

    # Instantiate model structure matching P300 / default dimensions
    model = BENDRClassification(
        targets=2, 
        samples=1000, 
        channels=20, 
        encoder_h=512, 
        contextualizer_hidden=3076
    )

    # Check for pre-trained weights and load them conditionally
    if os.path.exists(encoder_checkpoint) and os.path.exists(contextualizer_checkpoint):
        print(f"Found pre-trained weights in {checkpoint_dir}. Loading modules...")
        try:
            model.load_pretrained_modules(
                encoder_file=encoder_checkpoint,
                contextualizer_file=contextualizer_checkpoint,
                strict=False
            )
            print("Pre-trained weights loaded successfully.")
        except Exception as e:
            print(f"Error loading checkpoint modules: {e}")
    else:
        print(f"Notice: Pre-trained weight files ('encoder.pt' / 'contextualizer.pt') not found in '{checkpoint_dir}'.")
        print("Proceeding with model architecture structural smoke test (random initialization)...")

    model.to(device)
    model.eval()

    # Create synthetic EEG batch: (Batch Size, Channels, Time-samples)
    dummy_eeg = torch.randn(2, 20, 1000).to(device)
    print(f"Input EEG tensor shape: {dummy_eeg.shape}")

    # Run forward pass smoke test
    with torch.no_grad():
        output_features = model.features_forward(dummy_eeg)
        print(f"Model forward output features shape: {output_features.shape}")

    print("Smoke test execution completed successfully!")

if __name__ == "__main__":
    test_bendr_inference()
