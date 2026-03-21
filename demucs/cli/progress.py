# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from collections.abc import Callable

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .utils import console


def create_model_progress_bar() -> Progress:
    """
    Create a standardized progress bar for model operations.

    :return: Configured Rich Progress instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
        refresh_per_second=2,
        expand=True,
    )


def create_progress_callback(progress_bar: Progress, task: TaskID) -> Callable[[str, dict], None]:
    """
    Create a progress callback that updates a Rich progress bar.

    :param progress_bar: Rich Progress instance to update
    :param task: Task ID within the progress bar
    :return: Callback function for model download progress
    """

    def callback(event_type: str, data: dict):
        if event_type == "layer_start":
            progress_bar.update(
                task,
                description=f"[cyan]Downloading {data['model_name']}[/cyan] - Layer {data['layer_index']}/{data['total_layers']}",
            )
        elif event_type == "layer_progress":
            layer_base = (data["layer_index"] - 1) / data["total_layers"] * 100
            layer_progress = data["progress_percent"] / data["total_layers"]
            overall_progress = layer_base + layer_progress

            phase_text = ""
            if "phase" in data:
                phase_text = f" ({data['phase']})"

            progress_bar.update(
                task,
                completed=overall_progress,
                description=f"[cyan]Downloading {data['model_name']}[/cyan] - Layer {data['layer_index']}/{data['total_layers']}{phase_text}",
            )
        elif event_type == "layer_complete":
            if data.get("cached"):
                progress_bar.update(
                    task,
                    completed=(data["layer_index"] / data["total_layers"]) * 100,
                    description=f"[cyan]Downloading {data['model_name']}[/cyan] - Layer {data['layer_index']}/{data['total_layers']} (cached)",
                )
            else:
                progress_bar.update(
                    task,
                    completed=(data["layer_index"] / data["total_layers"]) * 100,
                    description=f"[cyan]Downloading {data['model_name']}[/cyan] - Layer {data['layer_index']}/{data['total_layers']} (complete)",
                )
        elif event_type == "download_complete":
            progress_bar.update(
                task,
                completed=100,
                description=f"[green]Downloaded {data['model_name']}[/green] - All {data['total_layers']} layers complete",
            )

    return callback


def create_file_progress_bar() -> Progress:
    """
    Create a progress bar optimized for processing multiple files.

    :return: Configured Rich Progress instance
    """
    return Progress(
        SpinnerColumn(finished_text="[green]✓[/green]"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
        refresh_per_second=4,
        expand=True,
    )


class FileProgressTracker:
    """
    Tracks separation progress across multiple files.
    """

    def __init__(self, total_files: int) -> None:
        """
        Initialize the file progress tracker.

        :param total_files: Total number of files to process
        """
        self.total_files = total_files
        self.completed_files = 0
        self.progress_bar = None
        self.current_task = None
        self.file_tasks = {}

    def __enter__(self) -> "FileProgressTracker":
        """
        Enter the progress tracker context.

        :return: This tracker instance
        """
        self.progress_bar = create_file_progress_bar()
        self.progress_bar.__enter__()
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object) -> None:
        """
        Exit the progress tracker context.

        :param exc_type: Exception type, if any
        :param exc_val: Exception value, if any
        :param exc_tb: Exception traceback, if any
        """
        if self.progress_bar:
            self.progress_bar.__exit__(exc_type, exc_val, exc_tb)

    def start_file(self, filename: str) -> TaskID:
        """
        Start processing a new file.

        :param filename: Name of the file being processed
        :return: Task ID for the progress bar entry
        """
        task_id = self.progress_bar.add_task(filename.strip(), total=100, completed=0)
        self.file_tasks[filename] = task_id
        return task_id

    def update_file_progress(self, filename: str, event_type: str, data: dict) -> None:
        """
        Update progress for a specific file.

        :param filename: Name of the file to update
        :param event_type: Type of progress event
        :param data: Event data dictionary
        """
        if filename not in self.file_tasks:
            return

        task_id = self.file_tasks[filename]

        if event_type == "processing_start":
            self.progress_bar.update(
                task_id,
                description=filename.strip(),
                total=data["total_chunks"],
                completed=0,
            )
        elif event_type == "chunk_complete":
            self.progress_bar.update(
                task_id,
                completed=data["completed_chunks"],
                description=filename.strip(),
            )
        elif event_type == "processing_complete":
            self.progress_bar.update(
                task_id, completed=data["total_chunks"], description=filename.strip()
            )
            self.completed_files += 1

    def error_file(self, filename: str, error_msg: str) -> None:
        """
        Mark a file as having an error.

        :param filename: Name of the file that errored
        :param error_msg: Error message to display
        """
        if filename not in self.file_tasks:
            return

        task_id = self.file_tasks[filename]
        self.progress_bar.update(
            task_id, completed=100, description=f"[red]✗[/red] {filename.strip()}"
        )
        self.completed_files += 1

    def create_audio_callback(self, filename: str) -> Callable[[str, dict], None]:
        """
        Create a callback for audio processing progress.

        :param filename: Name of the file being processed
        :return: Callback function for audio processing events
        """

        def callback(event_type: str, data: dict):
            self.update_file_progress(filename, event_type, data)

        return callback
