import logging
import shutil
from datetime import datetime
from pathlib import Path

class RemoteBackuper:
    """
    Handles the archiving and copying of experiment artifacts to a
    remote destination, like a mounted Google Drive, using pathlib.
    """
    def __init__(self, destination_dir: str | Path):
        """
        Initializes the backuper with the remote destination path.
        """
        # Ensure the destination is a Path object
        self.destination_dir = Path(destination_dir)
        self.destination_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"RemoteBackuper initialized. Destination: '{self.destination_dir}'")

    def backup_file(self, source_path: str | Path):
        """
        Copies a single file to the destination directory.
        """
        source = Path(source_path)
        if not source.exists():
            logging.error(f"Backup failed: Source file not found at '{source}'")
            return

        try:
            destination_path = self.destination_dir / source.name
            logging.info(f"Copying '{source}' to '{destination_path}'...")
            shutil.copyfile(source, destination_path)
            logging.info("✅ File backup successful.")
        except Exception as e:
            logging.error(f"❌ Error during file backup: {e}")

    def backup_directory_as_zip(self, source_dir: str | Path, archive_name_prefix: str):
        """
        Creates a timestamped zip archive of a source directory and copies
        it to the destination.
        """
        source = Path(source_dir)
        if not source.is_dir():
            logging.error(f"Backup failed: Source directory not found at '{source}'")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"{archive_name_prefix}_{timestamp}"
            
            logging.info(f"Creating archive '{archive_name}.zip' from '{source}'...")
            
            # Create the zip file locally first
            archive_path_str = shutil.make_archive(
                base_name=archive_name,
                format='zip',
                root_dir=source
            )
            #archive_path_str = shutil.make_archive(
                #base_name=archive_name,
                #format='zip',
                #root_dir=source.parent,
                #base_dir=source.name
            #)
            
            # Convert the returned string path to a Path object
            local_archive_path = Path(archive_path_str)

            # Copy the final zip file to the remote destination
            destination_zip_path = self.destination_dir / local_archive_path.name
            shutil.copyfile(local_archive_path, destination_zip_path)
            
            # Clean up the local zip file after copying
            local_archive_path.unlink()
            
            logging.info(f"✅ Directory backup successful. Archive saved to '{destination_zip_path}'")
        except Exception as e:
            logging.error(f"❌ Error during directory backup: {e}")
