"""
executor.py — DuckDB In-Memory SQL Execution Engine
=====================================================
The core innovation of this environment: instead of keyword-matching
heuristics, we ACTUALLY execute both the original and optimized queries
against realistic synthetic data and measure real performance differences.

Tables populated:
  users    — 10,000 rows
  orders   — 500,000 rows
  products —  1,000 rows
  events   — 1,000,000 rows
"""

import atexit
import os
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import duckdb

_instance: Optional["QueryExecutor"] = None
_lock = threading.Lock()


def _strip_terminators(query: str) -> str:
    """Trim surrounding whitespace and any trailing semicolons.

    Trailing ``;`` breaks queries that are wrapped in a subquery
    (``SELECT COUNT(*) FROM ({query}) t``) for correctness checksums.
    """
    return query.strip().rstrip(";").strip()


class QueryExecutor:
    """
    Runs SQL against an in-memory DuckDB database with realistic
    synthetic data.  Provides execution timing, result correctness
    checks, and EXPLAIN plans — all used by the reward function.
    """

    def __init__(self) -> None:
        # Build the tables once on a writable connection to a temp file, then
        # reopen the database read-only. Every query — trusted task SQL and
        # agent rewrites alike — runs against the read-only connection, so the
        # DuckDB engine itself rejects any write (no keyword heuristics, and DML
        # inside a CTE cannot slip through on any DuckDB version). A lock
        # serializes access because a single DuckDB connection is not safe for
        # concurrent use by the server's worker pool.
        self._dir = tempfile.mkdtemp(prefix="sql_optim_")
        self._path = os.path.join(self._dir, "sql_optim.duckdb")
        builder = duckdb.connect(self._path, config={"threads": "2"})
        self._build_tables(builder)
        builder.close()

        self.conn = duckdb.connect(self._path, read_only=True, config={"threads": "2"})
        self._exec_lock = threading.Lock()
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        """Close the connection and remove the temp database directory."""
        try:
            self.conn.close()
        except Exception:
            pass
        shutil.rmtree(self._dir, ignore_errors=True)

    # ── Schema Setup ─────────────────────────────────────────────────────

    def _build_tables(self, conn: "duckdb.DuckDBPyConnection") -> None:
        """Create and populate all four synthetic tables on *conn*."""

        # users — 10k rows
        conn.execute("""
            CREATE TABLE users AS
            SELECT
                i                                                      AS id,
                'u' || i || '@mail.com'                                AS email,
                CASE i % 3
                    WHEN 0 THEN 'premium'
                    WHEN 1 THEN 'free'
                    ELSE 'enterprise' END                              AS tier,
                CASE i % 5
                    WHEN 0 THEN 'US'   WHEN 1 THEN 'EU'
                    WHEN 2 THEN 'IN'   WHEN 3 THEN 'UK'
                    ELSE 'AU' END                                      AS region,
                CASE i % 2 WHEN 0 THEN 'premium' ELSE 'basic' END     AS plan,
                DATE '2020-01-01' + CAST(i AS INTEGER)                 AS created_at
            FROM generate_series(1, 10000) t(i)
        """)

        # orders — 500k rows
        conn.execute("""
            CREATE TABLE orders AS
            SELECT
                i                                                      AS id,
                1 + (i % 10000)                                        AS customer_id,
                (i % 100) + 1                                          AS product_id,
                CASE i % 4
                    WHEN 0 THEN 'completed'  WHEN 1 THEN 'pending'
                    WHEN 2 THEN 'cancelled'  ELSE 'shipped' END        AS status,
                ROUND((i % 1000) * 1.5 + 49.99, 2)                   AS total,
                DATE '2023-01-01' + CAST(i % 730 AS INTEGER)          AS created_at
            FROM generate_series(1, 500000) t(i)
        """)

        # products — 1k rows
        conn.execute("""
            CREATE TABLE products AS
            SELECT
                i                                                      AS id,
                'Product_' || i                                        AS name,
                CASE i % 5
                    WHEN 0 THEN 'Electronics'  WHEN 1 THEN 'Clothing'
                    WHEN 2 THEN 'Food'         WHEN 3 THEN 'Books'
                    ELSE 'Sports' END                                  AS category,
                ROUND((i % 500) + 9.99, 2)                            AS price
            FROM generate_series(1, 1000) t(i)
        """)

        # events — 1M rows
        conn.execute("""
            CREATE TABLE events AS
            SELECT
                i                                                      AS id,
                1 + (i % 10000)                                        AS user_id,
                'sess_' || (i % 50000)                                 AS session_id,
                CASE i % 6
                    WHEN 0 THEN 'purchase'  WHEN 1 THEN 'view'
                    WHEN 2 THEN 'click'     WHEN 3 THEN 'signup'
                    WHEN 4 THEN 'logout'    ELSE 'search' END          AS event_type,
                DATE '2024-01-01' + CAST(i % 365 AS INTEGER)          AS occurred_at
            FROM generate_series(1, 1000000) t(i)
        """)

    # ── Execution helpers ─────────────────────────────────────────────────

    def _run(
        self, query: str, runs: int = 3
    ) -> Tuple[float, Optional[List], Optional[str]]:
        """
        Execute *query* up to *runs* times on the read-only connection.
        Returns (median_ms, rows, error_or_None). A write raises and is returned
        as the error string.
        """
        timings: List[float] = []
        rows: Optional[List] = None

        with self._exec_lock:
            for _ in range(runs):
                try:
                    t0 = time.perf_counter()
                    rows = self.conn.execute(query).fetchall()
                    timings.append((time.perf_counter() - t0) * 1000.0)
                except Exception as exc:
                    return 99_999.0, None, str(exc)

        timings.sort()
        return round(timings[len(timings) // 2], 3), rows, None

    def _checksum(
        self, query: str
    ) -> Tuple[Optional[int], Optional[int], Optional[str]]:
        """
        Compute a deterministic (row-order-independent) checksum.
        Returns (row_count, checksum, error).

        BIT_XOR is commutative+associative — order-independent fingerprint.
        Falls back to count-only if the DuckDB version doesn't support the function.
        """
        # Try BIT_XOR of a numeric hash (portable across DuckDB versions)
        for sql_template in [
            # Option 1: BIT_XOR of md5 prefix cast to integer
            (
                "SELECT COUNT(*) AS cnt, "
                "BIT_XOR(CAST(('0x' || LEFT(md5(CAST(t AS VARCHAR)), 15)) AS UBIGINT)) AS chk "
                "FROM ({query}) t"
            ),
            # Option 2: sum of hash (order-independent since sum is commutative)
            (
                "SELECT COUNT(*) AS cnt, "
                "SUM(hash(CAST(t AS VARCHAR)) % 9999999999) AS chk "
                "FROM ({query}) t"
            ),
        ]:
            try:
                wrapped = sql_template.format(query=query)
                with self._exec_lock:
                    result = self.conn.execute(wrapped).fetchone()
                return result[0], result[1], None
            except Exception:
                continue
        # Final fallback: count only
        try:
            with self._exec_lock:
                cnt = self.conn.execute(f"SELECT COUNT(*) FROM ({query}) t").fetchone()[
                    0
                ]
            return cnt, None, None
        except Exception as exc:
            return None, None, str(exc)

    # ── Public API ────────────────────────────────────────────────────────

    def compare(self, original: str, optimized: str) -> Dict[str, Any]:
        """
        Execute both queries, measure real timing, check correctness.

        Returns a dict with:
          original_ms, optimized_ms, speedup,
          results_match, original_rows, optimized_rows,
          original_error, optimized_error, verdict
        """
        # Normalize (drop trailing ``;``) so both direct execution and the
        # subquery-wrapped checksum are valid.
        original = _strip_terminators(original)
        optimized = _strip_terminators(optimized)

        orig_ms, orig_rows, orig_err = self._run(original)

        # The optimized query is agent-authored, but the read-only connection is
        # the guarantee: any write (DDL/DML, in a CTE, or after a `;`) is rejected
        # by the DuckDB engine and surfaces as `optimized_error`. No structural
        # pre-check is needed, which also avoids false rejections of valid
        # rewrites (leading comments, a `;` inside a string literal, etc.).
        opt_ms, opt_rows, opt_err = self._run(optimized)

        # ── Correctness: do both queries return the same data? ────────
        # Use a DuckDB-level checksum (order-independent) to avoid
        # false negatives from non-deterministic row ordering in parallel
        # window function queries on large tables.
        results_match = False
        if orig_rows is not None and opt_rows is not None:
            try:
                if len(orig_rows) != len(opt_rows):
                    results_match = False
                elif len(orig_rows) == 0:
                    results_match = True
                elif len(orig_rows) <= 50_000:
                    # Small/medium: full sorted comparison (precise)
                    orig_s = sorted(str(r) for r in orig_rows)
                    opt_s = sorted(str(r) for r in opt_rows)
                    results_match = orig_s == opt_s
                else:
                    # Large result sets: use SQL-level hash checksum
                    # (deterministic regardless of row ordering / thread count)
                    o_cnt, o_chk, o_err2 = self._checksum(original)
                    p_cnt, p_chk, p_err2 = self._checksum(optimized)
                    if o_err2 or p_err2:
                        # Checksum failed — fall back to row count
                        results_match = len(orig_rows) == len(opt_rows)
                    else:
                        results_match = (o_cnt == p_cnt) and (o_chk == p_chk)
            except Exception:
                results_match = len(orig_rows) == len(opt_rows)

        # ── Speedup ratio ─────────────────────────────────────────────
        speedup = 1.0
        if opt_ms > 0 and orig_ms < 90_000:
            speedup = round(orig_ms / opt_ms, 3)

        # ── Human-readable verdict ────────────────────────────────────
        if opt_err:
            verdict = f"[FAIL] Optimized query error: {opt_err[:120]}"
        elif results_match and speedup >= 2.0:
            verdict = f"[OK] {speedup:.1f}x faster with correct results"
        elif results_match and speedup >= 1.0:
            verdict = (
                f"[WARN] Correct results but only {speedup:.1f}x speedup -- dig deeper"
            )
        elif not results_match and speedup >= 2.0:
            verdict = (
                f"[WARN] {speedup:.1f}x faster but results don't match -- fix the logic"
            )
        else:
            verdict = f"[FAIL] {speedup:.1f}x -- no meaningful improvement"

        return {
            "original_ms": orig_ms,
            "optimized_ms": opt_ms,
            "speedup": speedup,
            "results_match": results_match,
            "original_rows": len(orig_rows) if orig_rows is not None else 0,
            "optimized_rows": len(opt_rows) if opt_rows is not None else 0,
            "original_error": orig_err,
            "optimized_error": opt_err,
            "verdict": verdict,
        }

    def explain(self, query: str) -> str:
        """Return EXPLAIN output for a query."""
        try:
            with self._exec_lock:
                rows = self.conn.execute(
                    f"EXPLAIN {_strip_terminators(query)}"
                ).fetchall()
            return "\n".join(str(r[1]) for r in rows)
        except Exception as exc:
            return f"EXPLAIN error: {exc}"

    @property
    def table_stats(self) -> Dict[str, int]:
        tables = ["users", "orders", "products", "events"]
        with self._exec_lock:
            return {
                t: self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in tables
            }


# ── Singleton accessor ────────────────────────────────────────────────────


def get_executor() -> QueryExecutor:
    """Return the process-level singleton (lazy init, thread-safe)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = QueryExecutor()
    return _instance
