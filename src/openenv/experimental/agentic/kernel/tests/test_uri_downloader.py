# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for URIDownloader."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from agentic.kernel.core.storage.blob import LocalBlobStore
from agentic.kernel.core.storage.uri import URIDownloader


@pytest.fixture
def blob_store(tmp_path: Path) -> LocalBlobStore:
    return LocalBlobStore(tmp_path / "blobs")


@pytest.fixture
def downloader(blob_store: LocalBlobStore) -> URIDownloader:
    return URIDownloader(blob_store=blob_store)


@pytest.fixture
def downloader_no_blob() -> URIDownloader:
    return URIDownloader(blob_store=None)


# ── blob:// Tests ────────────────────────────────────────────────────


class TestBlobDownload:
    @pytest.mark.asyncio
    async def test_download_blob_file(
        self,
        downloader: URIDownloader,
        blob_store: LocalBlobStore,
        tmp_path: Path,
    ) -> None:
        # Upload a file to blob store
        src = tmp_path / "hello.txt"
        src.write_text("hello world")
        uri = blob_store.upload(src)

        # Download it
        dest = tmp_path / "output"
        result = await downloader.download(uri, dest)

        assert result == dest
        assert dest.exists()

    @pytest.mark.asyncio
    async def test_download_blob_directory(
        self,
        downloader: URIDownloader,
        blob_store: LocalBlobStore,
        tmp_path: Path,
    ) -> None:
        # Upload a directory to blob store
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "a.py").write_text("x = 1")
        (src_dir / "b.py").write_text("y = 2")
        uri = blob_store.upload_dir(src_dir)

        # Download it
        dest = tmp_path / "output"
        await downloader.download(uri, dest)

        assert (dest / "a.py").read_text() == "x = 1"
        assert (dest / "b.py").read_text() == "y = 2"

    @pytest.mark.asyncio
    async def test_download_blob_without_blob_store_raises(
        self,
        downloader_no_blob: URIDownloader,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="blob_store required"):
            await downloader_no_blob.download("blob://abc123", tmp_path / "out")

    @pytest.mark.asyncio
    async def test_download_blob_nonexistent_raises(
        self,
        downloader: URIDownloader,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(KeyError):
            await downloader.download("blob://nonexistent", tmp_path / "out")


# ── file:// Tests ────────────────────────────────────────────────────


class TestFileDownload:
    @pytest.mark.asyncio
    async def test_download_file_single(
        self,
        downloader: URIDownloader,
        tmp_path: Path,
    ) -> None:
        src = tmp_path / "source.txt"
        src.write_text("file content")

        dest = tmp_path / "dest.txt"
        result = await downloader.download(f"file://{src}", dest)

        assert result == dest
        assert dest.read_text() == "file content"

    @pytest.mark.asyncio
    async def test_download_file_directory(
        self,
        downloader: URIDownloader,
        tmp_path: Path,
    ) -> None:
        src_dir = tmp_path / "srcdir"
        src_dir.mkdir()
        (src_dir / "f1.txt").write_text("file 1")
        sub = src_dir / "sub"
        sub.mkdir()
        (sub / "f2.txt").write_text("file 2")

        dest = tmp_path / "destdir"
        await downloader.download(f"file://{src_dir}", dest)

        assert (dest / "f1.txt").read_text() == "file 1"
        assert (dest / "sub" / "f2.txt").read_text() == "file 2"

    @pytest.mark.asyncio
    async def test_download_file_nonexistent_raises(
        self,
        downloader: URIDownloader,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(FileNotFoundError):
            await downloader.download("file:///nonexistent/path", tmp_path / "out")


# ── s3:// Tests (mocked) ────────────────────────────────────────────


class TestS3Download:
    @pytest.mark.asyncio
    async def test_download_s3_success(
        self,
        downloader: URIDownloader,
        tmp_path: Path,
    ) -> None:
        dest = tmp_path / "s3output"

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await downloader.download("s3://my-bucket/blobs/abc123", dest)

        assert result == dest

    @pytest.mark.asyncio
    async def test_download_s3_fallback_to_non_recursive(
        self,
        downloader: URIDownloader,
        tmp_path: Path,
    ) -> None:
        """When --recursive fails, it retries without --recursive."""
        dest = tmp_path / "s3output"

        call_count = 0

        async def mock_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock = AsyncMock()
            if call_count == 1:
                # First call (with --recursive) fails
                mock.returncode = 1
                mock.communicate = AsyncMock(return_value=(b"", b"not a dir"))
            else:
                # Second call (without --recursive) succeeds
                mock.returncode = 0
                mock.communicate = AsyncMock(return_value=(b"ok", b""))
            return mock

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            result = await downloader.download("s3://my-bucket/blobs/abc123", dest)

        assert result == dest
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_download_s3_failure_raises(
        self,
        downloader: URIDownloader,
        tmp_path: Path,
    ) -> None:
        dest = tmp_path / "s3output"

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"access denied"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="aws s3 cp failed"):
                await downloader.download("s3://my-bucket/blobs/abc123", dest)


# ── Unsupported Scheme Tests ─────────────────────────────────────────


class TestUnsupportedScheme:
    @pytest.mark.asyncio
    async def test_unsupported_scheme_raises(
        self,
        downloader: URIDownloader,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="Unsupported URI scheme"):
            await downloader.download("ftp://example.com/file", tmp_path / "out")

    @pytest.mark.asyncio
    async def test_no_scheme_raises(
        self,
        downloader: URIDownloader,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="Unsupported URI scheme"):
            await downloader.download("/just/a/path", tmp_path / "out")
