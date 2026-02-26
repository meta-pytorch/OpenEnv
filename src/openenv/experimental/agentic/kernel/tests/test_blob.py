# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for LocalBlobStore."""

from pathlib import Path

import pytest
from agentic.kernel.core.storage.blob import LocalBlobStore


@pytest.fixture
def blob_store(tmp_path: Path) -> LocalBlobStore:
    return LocalBlobStore(tmp_path / "blobs")


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    f = tmp_path / "hello.txt"
    f.write_text("hello world")
    return f


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    d = tmp_path / "mydir"
    d.mkdir()
    (d / "a.txt").write_text("file a")
    sub = d / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("file b")
    return d


class TestUploadFile:
    def test_returns_blob_uri(self, blob_store: LocalBlobStore, sample_file: Path):
        uri = blob_store.upload(sample_file)
        assert uri.startswith("blob://")

    def test_file_content_persisted(
        self, blob_store: LocalBlobStore, sample_file: Path
    ):
        uri = blob_store.upload(sample_file)
        stored_path = blob_store.get_path(uri)
        stored_file = stored_path / sample_file.name
        assert stored_file.read_text() == "hello world"

    def test_same_content_same_hash(self, blob_store: LocalBlobStore, tmp_path: Path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("identical")
        f2.write_text("identical")
        uri1 = blob_store.upload(f1)
        uri2 = blob_store.upload(f2)
        # Same content -> same hash prefix (though filenames differ)
        assert uri1 == uri2

    def test_different_content_different_hash(
        self, blob_store: LocalBlobStore, tmp_path: Path
    ):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content a")
        f2.write_text("content b")
        assert blob_store.upload(f1) != blob_store.upload(f2)

    def test_upload_nonexistent_file_raises(self, blob_store: LocalBlobStore):
        with pytest.raises(FileNotFoundError):
            blob_store.upload("/nonexistent/file.txt")


class TestUploadDir:
    def test_returns_blob_uri(self, blob_store: LocalBlobStore, sample_dir: Path):
        uri = blob_store.upload_dir(sample_dir)
        assert uri.startswith("blob://")

    def test_directory_structure_preserved(
        self, blob_store: LocalBlobStore, sample_dir: Path
    ):
        uri = blob_store.upload_dir(sample_dir)
        stored = blob_store.get_path(uri)
        assert (stored / "a.txt").read_text() == "file a"
        assert (stored / "sub" / "b.txt").read_text() == "file b"

    def test_upload_not_a_directory_raises(
        self, blob_store: LocalBlobStore, sample_file: Path
    ):
        with pytest.raises(NotADirectoryError):
            blob_store.upload_dir(sample_file)


class TestGetPathAndExists:
    def test_exists_after_upload(self, blob_store: LocalBlobStore, sample_file: Path):
        uri = blob_store.upload(sample_file)
        assert blob_store.exists(uri)

    def test_not_exists_for_unknown_uri(self, blob_store: LocalBlobStore):
        assert not blob_store.exists("blob://deadbeef")

    def test_get_path_raises_for_unknown(self, blob_store: LocalBlobStore):
        with pytest.raises(KeyError):
            blob_store.get_path("blob://deadbeef")
