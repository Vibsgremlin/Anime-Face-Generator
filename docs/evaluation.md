# Evaluation Logs and Outputs

## What was evaluated
- Dataset loading from a local image directory
- Adversarial training loop for generator and discriminator
- Epoch-wise sample export
- Generator checkpoint creation

## Example training-style output
```text
Loaded 12000 images from local dataset
Starting Training...
Epoch [1/30] | Loss_D: 1.2043 | Loss_G: 2.8114
Epoch [2/30] | Loss_D: 0.9821 | Loss_G: 3.1048
Generator model saved as generator.pth
```

## Notes
- No FID, IS, or human quality rating report is committed in the repo.
- Output quality is tracked visually through saved sample grids.
