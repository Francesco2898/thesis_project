from pathlib import Path
import re
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def sanitize_filename(name: str) -> str:
    """Convert TensorBoard tag/run names into safe filenames."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def export_scalars_from_run(run_dir: Path, output_dir: Path) -> None:
    """Export all scalar tags from one TensorBoard run directory."""
    try:
        event_acc = EventAccumulator(str(run_dir))
        event_acc.Reload()
    except Exception as exc:
        print(f"[WARNING] Could not read {run_dir}: {exc}")
        return

    scalar_tags = event_acc.Tags().get("scalars", [])

    if not scalar_tags:
        return

    relative_run_name = sanitize_filename(run_dir.name)

    for tag in scalar_tags:
        events = event_acc.Scalars(tag)

        data = [
            {
                "wall_time": event.wall_time,
                "step": event.step,
                "value": event.value,
                "tag": tag,
                "run": str(run_dir),
            }
            for event in events
        ]

        df = pd.DataFrame(data)

        tag_name = sanitize_filename(tag)
        output_file = output_dir / f"{relative_run_name}__{tag_name}.csv"
        df.to_csv(output_file, index=False)

        print(f"[INFO] Saved {output_file}")


def main():
    # CHANGE THIS PATH
    log_root = Path(r"C:\Users\franc\thesis_project\thesis_project\runs\pick_place")

    # CHANGE THIS PATH IF NEEDED
    output_dir = Path(r"C:\Users\franc\thesis_project\exported_tensorboard_csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Search all subfolders containing TensorBoard event files
    event_files = list(log_root.rglob("events.out.tfevents.*"))

    if not event_files:
        print(f"[ERROR] No TensorBoard event files found in {log_root}")
        return

    run_dirs = sorted({event_file.parent for event_file in event_files})

    print(f"[INFO] Found {len(run_dirs)} run directories.")

    for run_dir in run_dirs:
        export_scalars_from_run(run_dir, output_dir)


if __name__ == "__main__":
    main()