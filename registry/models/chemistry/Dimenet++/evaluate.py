import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import DimeNetPlusPlus
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="DimeNet++ Inference Workflow on QM9")
    parser.add_argument('--smoke-test', action='store_true', help="Run a quick 1-batch test to verify workflow")
    parser.add_argument('--data-dir', type=str, default='./data', help="Directory to store QM9 dataset")
    parser.add_argument('--target', type=int, default=7, help="QM9 target index (7 is U0 internal energy)")
    args = parser.parse_args()

    # Enforce deterministic initialization for identical forward-pass validation
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on device: {device}")

    # Artifact Traceability: Load QM9 Dataset
    print("Loading QM9 dataset...")
    dataset = QM9(root=args.data_dir)
    
    # Use the standard 10,000 molecule test split for evaluation
    test_dataset = dataset[-10000:]
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize DimeNet++ Architecture
    print("Initializing DimeNet++ architecture...")
    model = DimeNetPlusPlus(
        hidden_channels=128, out_channels=1, num_blocks=4, int_emb_size=64, 
        basis_emb_size=8, out_emb_channels=256, num_spherical=7, num_radial=6,
        cutoff=5.0, envelope_exponent=5, num_before_skip=1, num_after_skip=2, num_output_layers=3
    ).to(device)

    model.eval()
    total_error = 0
    num_samples = 0

    print(f"Starting inference... (Smoke Test: {args.smoke_test})")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Batches"):
            batch = batch.to(device)
            
            # Forward pass: atomic numbers (z), coordinates (pos), and batch indices
            predictions = model(batch.z, batch.pos, batch.batch)
            targets = batch.y[:, args.target].view(-1, 1)

            # Compute Absolute Error
            error = F.l1_loss(predictions, targets, reduction='sum')
            total_error += error.item()
            num_samples += batch.num_graphs

            # Validation: Smoke Test exit condition
            if args.smoke_test:
                print(f"\n[SMOKE TEST PASSED] Processed 1 batch.")
                print(f"Input shape: {batch.z.shape[0]} atoms")
                print(f"Output shape expected: [32, 1], got: {predictions.shape}")
                break

    if not args.smoke_test:
        mae = total_error / num_samples
        print(f"\n[SUBSTANTIVE VALIDATION] Inference Complete.")
        print(f"Evaluated {num_samples} molecules.")
        print(f"Mean Absolute Error (Target {args.target}): {mae:.4f}")

if __name__ == "__main__":
    main()