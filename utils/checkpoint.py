from pathlib import Path
from datetime import datetime

def find_lastest_checkpoint(folder):
    # Filter for directories only
    folders = [f for f in Path(folder).glob("*") if f.is_dir()]
    
    # Parse folder name as time and sort by folder name
    folders.sort(key=lambda x: datetime.strptime(x.name, "%Y-%m-%d_%H-%M"), reverse=True)

    for subfolder in folders:  # Renamed to avoid shadowing
        checkpoints = list(Path(subfolder).glob("*.pth"))
        if len(checkpoints) > 0:
            # Sort by numeric value extracted from filename
            checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1]))
            print(f"Found checkpoint: {checkpoints[-1]}")
            latest_checkpoint = checkpoints[-1]

            # Get the checkpoint number without extension
            checkpoint_number = checkpoints[-1].stem.split("_")[-1]
            return latest_checkpoint, int(checkpoint_number)
    
    raise ValueError("No checkpoint found")

if __name__ == "__main__":
    print(find_lastest_checkpoint("checkpoints/decomposer"))