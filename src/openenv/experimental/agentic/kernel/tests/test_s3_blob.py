# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for S3BlobStore (all AWS CLI calls are mocked)."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from agentic.kernel.core.storage.s3_blob import S3BlobStore


@pytest.fixture
def s3_store() -> S3BlobStore:
    return S3BlobStore(bucket="test-bucket", prefix="blobs")


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


# ── Upload Tests ─────────────────────────────────────────────────────


class TestUpload:
    @pytest.mark.asyncio
    async def test_upload_returns_s3_uri(
        self, s3_store: S3BlobStore, sample_file: Path
    ) -> None:
        # Mock exists (not found) and cp (success)
        mock_ls = AsyncMock()
        mock_ls.returncode = 1
        mock_ls.communicate = AsyncMock(return_value=(b"", b""))

        mock_cp = AsyncMock()
        mock_cp.returncode = 0
        mock_cp.communicate = AsyncMock(return_value=(b"ok", b""))

        call_count = 0

        async def mock_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if args[1] == "s3" and args[2] == "ls":
                return mock_ls
            return mock_cp

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            uri = await s3_store.upload(sample_file)

        assert uri.startswith("s3://test-bucket/blobs/")
        assert len(uri.split("/")[-1]) == 64  # SHA-256 hex digest

    @pytest.mark.asyncio
    async def test_upload_skips_if_exists(
        self, s3_store: S3BlobStore, sample_file: Path
    ) -> None:
        # Mock exists (found)
        mock_ls = AsyncMock()
        mock_ls.returncode = 0
        mock_ls.communicate = AsyncMock(return_value=(b"blob", b""))

        cp_called = False

        async def mock_exec(*args, **kwargs):
            nonlocal cp_called
            if args[1] == "s3" and args[2] == "ls":
                return mock_ls
            cp_called = True
            mock = AsyncMock()
            mock.returncode = 0
            mock.communicate = AsyncMock(return_value=(b"ok", b""))
            return mock

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            uri = await s3_store.upload(sample_file)

        assert uri.startswith("s3://test-bucket/blobs/")
        assert not cp_called

    @pytest.mark.asyncio
    async def test_upload_nonexistent_file_raises(self, s3_store: S3BlobStore) -> None:
        with pytest.raises(FileNotFoundError):
            await s3_store.upload("/nonexistent/file.txt")

    @pytest.mark.asyncio
    async def test_upload_cp_failure_raises(
        self, s3_store: S3BlobStore, sample_file: Path
    ) -> None:
        mock_ls = AsyncMock()
        mock_ls.returncode = 1  # doesn't exist
        mock_ls.communicate = AsyncMock(return_value=(b"", b""))

        mock_cp = AsyncMock()
        mock_cp.returncode = 1
        mock_cp.communicate = AsyncMock(return_value=(b"", b"access denied"))

        async def mock_exec(*args, **kwargs):
            if args[1] == "s3" and args[2] == "ls":
                return mock_ls
            return mock_cp

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            with pytest.raises(RuntimeError, match="aws s3 cp failed"):
                await s3_store.upload(sample_file)


# ── Upload Dir Tests ─────────────────────────────────────────────────


class TestUploadDir:
    @pytest.mark.asyncio
    async def test_upload_dir_returns_s3_uri(
        self, s3_store: S3BlobStore, sample_dir: Path
    ) -> None:
        mock_ls = AsyncMock()
        mock_ls.returncode = 1
        mock_ls.communicate = AsyncMock(return_value=(b"", b""))

        mock_cp = AsyncMock()
        mock_cp.returncode = 0
        mock_cp.communicate = AsyncMock(return_value=(b"ok", b""))

        async def mock_exec(*args, **kwargs):
            if args[1] == "s3" and args[2] == "ls":
                return mock_ls
            return mock_cp

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            uri = await s3_store.upload_dir(sample_dir)

        assert uri.startswith("s3://test-bucket/blobs/")

    @pytest.mark.asyncio
    async def test_upload_dir_not_a_directory_raises(
        self, s3_store: S3BlobStore, sample_file: Path
    ) -> None:
        with pytest.raises(NotADirectoryError):
            await s3_store.upload_dir(sample_file)

    @pytest.mark.asyncio
    async def test_upload_dir_deterministic_hash(
        self, s3_store: S3BlobStore, sample_dir: Path
    ) -> None:
        """Same directory contents should produce the same URI."""
        mock_ls = AsyncMock()
        mock_ls.returncode = 1
        mock_ls.communicate = AsyncMock(return_value=(b"", b""))

        mock_cp = AsyncMock()
        mock_cp.returncode = 0
        mock_cp.communicate = AsyncMock(return_value=(b"ok", b""))

        async def mock_exec(*args, **kwargs):
            if args[1] == "s3" and args[2] == "ls":
                return mock_ls
            return mock_cp

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            uri1 = await s3_store.upload_dir(sample_dir)
            uri2 = await s3_store.upload_dir(sample_dir)

        assert uri1 == uri2


# ── Exists Tests ─────────────────────────────────────────────────────


class TestExists:
    @pytest.mark.asyncio
    async def test_exists_true(self, s3_store: S3BlobStore) -> None:
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"blob", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            assert await s3_store.exists("s3://test-bucket/blobs/abc123") is True

    @pytest.mark.asyncio
    async def test_exists_false(self, s3_store: S3BlobStore) -> None:
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            assert await s3_store.exists("s3://test-bucket/blobs/abc123") is False

    @pytest.mark.asyncio
    async def test_exists_invalid_uri_raises(self, s3_store: S3BlobStore) -> None:
        with pytest.raises(ValueError, match="Not an s3:// URI"):
            await s3_store.exists("blob://abc123")


# ── Get URL Tests ────────────────────────────────────────────────────


class TestGetUrl:
    @pytest.mark.asyncio
    async def test_get_url(self, s3_store: S3BlobStore) -> None:
        url = await s3_store.get_url("s3://test-bucket/blobs/abc123")
        assert url == "https://test-bucket.s3.amazonaws.com/blobs/abc123"

    @pytest.mark.asyncio
    async def test_get_url_invalid_uri_raises(self, s3_store: S3BlobStore) -> None:
        with pytest.raises(ValueError, match="Not an s3:// URI"):
            await s3_store.get_url("blob://abc123")
