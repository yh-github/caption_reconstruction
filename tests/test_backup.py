from pathlib import Path
import zipfile

# The class we are testing
from backup import RemoteBackuper

def test_remote_backuper_backup_file(tmp_path: Path):
    """
    Tests that the backuper can correctly copy a single file.
    'tmp_path' is a pytest fixture that provides a temporary directory.
    """
    # Arrange
    # 1. Create temporary source and destination directories inside tmp_path
    source_dir = tmp_path / "source"
    dest_dir = tmp_path / "destination"
    source_dir.mkdir()
    dest_dir.mkdir()

    # 2. Create a dummy source file
    source_file = source_dir / "test_log.log"
    source_file.write_text("This is a test log.")

    # 3. Initialize our backuper
    backuper = RemoteBackuper(destination_dir=dest_dir)

    # Act
    # 4. Call the method we want to test
    backuper.backup_file(source_file)

    # Assert
    # 5. Check that the file was copied correctly
    expected_destination_file = dest_dir / "test_log.log"
    assert expected_destination_file.exists()
    assert expected_destination_file.read_text() == "This is a test log."

def test_remote_backuper_backup_directory_as_zip(tmp_path: Path):
    """
    Tests that the backuper can correctly archive a directory and
    copy the resulting zip file.
    """
    # Arrange
    source_dir = tmp_path / "source_mlruns"
    dest_dir = tmp_path / "destination_zips"
    source_dir.mkdir()
    dest_dir.mkdir()

    # Create some dummy files inside the source directory
    (source_dir / "file1.txt").write_text("file1")
    (source_dir / "file2.txt").write_text("file2")

    backuper = RemoteBackuper(destination_dir=dest_dir)

    # Act
    backuper.backup_directory_as_zip(
        source_dir=source_dir,
        archive_name_prefix="test_archive"
    )

    # Assert
    # 1. Check that a zip file was created in the destination
    # We use list() to find the single zip file in the directory
    zip_files = list(dest_dir.glob("test_archive_*.zip"))
    assert len(zip_files) == 1
    
    # 2. (Optional but recommended) Check the contents of the zip file
    zip_path = zip_files[0]
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Check that the files inside the zip have the correct names
        zipped_files = zf.namelist()
        assert "file1.txt" in zipped_files
        assert "file2.txt" in zipped_files
