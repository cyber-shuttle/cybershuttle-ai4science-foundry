set -e
mkdir -p data models outputs logs
echo "==> Downloading Darcy Flow 2D benchmark test dataset..."
# Direct mirror of the standard Darcy 64x64 test split from neuraloperator
curl -sSL -o data/darcy_test_64.pt "https://raw.githubusercontent.com/neuraloperator/neuraloperator/main/tests/test_data/darcy_test_64.pt"
echo "==> Verifying data artifact..."
test -s data/darcy_test_64.pt || { echo "ERROR: Darcy dataset download failed or file empty"; exit 1; }
echo "==> Artifact download complete."
ls -lh data/
