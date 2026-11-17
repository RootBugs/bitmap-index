"""bitmap_index — bit-sliced, WAH-compressed bitmap indexes over CSV tables.

Build bit-sliced indexes for the columns of a CSV file and answer boolean
queries *without materializing rows*:

    >>> import bitmap_index
    >>> eng = bitmap_index.BitmapIndex()
    >>> eng.add_column("age", [21, 45, 33])
    >>> eng.add_column("city", ["a", "b", "a"])
    >>> eng.count("age >= 30 AND city = 'a'")
    1

Storage layer
-------------
Bitmaps are stored as **WAH-style** run-length words: the bit stream is cut
into 31-bit chunks; a chunk is either a literal (flag bit clear) or a fill
word (flag set + fill value + repeat count) covering runs of uniform chunks.
Logical operations run over decoded chunk streams, and ``count()`` is answered
directly from the compressed words — rows are never enumerated to answer
``COUNT``.

Index layer
-----------
- *string* columns: one equality bitmap per distinct value; ranges combine the
  distinct values in lexicographic order via bisect.
- *int* columns: equality bitmaps **plus** bit-slices (one bitmap per bit of
  ``value - min``).  ``col < x`` uses the classic bit-slice comparison
  identity::

      s < t  ==  OR over set bits i of t:  ( AND over j > i: s_j == t_j ) AND s_i == 0

Query language
--------------
``= == != < <= > >= IN ( ... ) NOT AND OR`` over parenthesised expressions,
``AND`` binding tighter than ``OR``, ``NOT`` tightest of all.  Strings are
quoted with ``'`` or ``"``; unquoted words are string literals.
"""

from __future__ import annotations

import bisect
import csv
import json
import os
import struct
from typing import Iterable, Iterator, Sequence

__all__ = [
    "Bitmap",
    "BitmapIndex",
    "ColumnIndex",
    "IntColumnIndex",
    "StrColumnIndex",
    "QueryError",
    "QueryResult",
]

_CHUNK = 31
_MASK31 = (1 << _CHUNK) - 1
_FILL_FLAG = 1 << 31
_FILL_VAL = 1 << 30
_COUNT_MASK = (1 << 30) - 1
_MAGIC = b"BIDX1"


class QueryError(Exception):
    """A query could not be parsed or evaluated."""

    def __init__(self, message: str, position: int | None = None) -> None:
        self.position = position
        if position is not None:
            message = f"{message} (at position {position})"
        super().__init__(message)


# ---------------------------------------------------------------------------
# WAH-style compressed bitmap
# ---------------------------------------------------------------------------


def _encode_chunks(chunks: Iterable[int]) -> tuple[int, ...]:
    """Encode 31-bit chunk values as literal / fill words (WAH style)."""
    words: list[int] = []
    fill_val: int | None = None
    fill_len = 0

    def flush() -> None:
        nonlocal fill_val, fill_len
        if fill_len:
            words.append(
                _FILL_FLAG | fill_len | (_FILL_VAL if fill_val == _MASK31 else 0)
            )
            fill_val = None
            fill_len = 0

    for chunk in chunks:
        if chunk == 0 or chunk == _MASK31:
            if fill_val is not None and chunk == fill_val and fill_len < _COUNT_MASK:
                fill_len += 1
            else:
                flush()
                fill_val = chunk
                fill_len = 1
        else:
            flush()
            words.append(chunk)
    flush()
    return tuple(words)


class Bitmap:
    """An immutable bit vector compressed with WAH-style run-length encoding."""

    __slots__ = ("_words", "_length")

    def __init__(self, words: Sequence[int], length: int) -> None:
        if length < 0:
            raise ValueError("length must be >= 0")
        self._words = tuple(int(w) & 0xFFFFFFFF for w in words)
        self._length = length

    # -- construction ------------------------------------------------------

    @classmethod
    def from_positions(cls, length: int, positions: Iterable[int]) -> "Bitmap":
        """Build the bitmap with exactly the given bit positions set."""
        if length < 0:
            raise ValueError("length must be >= 0")
        data = bytearray((length + 7) // 8)
        for pos in positions:
            if not 0 <= pos < length:
                raise ValueError(f"position {pos} outside [0, {length})")
            data[pos >> 3] |= 1 << (pos & 7)
        return cls._from_bit_bytes(length, data)

    @classmethod
    def _from_bit_bytes(cls, length: int, data: bytes | bytearray) -> "Bitmap":
        def chunks() -> Iterator[int]:
            for base in range(0, length, _CHUNK):
                value = 0
                for k in range(_CHUNK):
                    pos = base + k
                    if pos >= length:
                        break
                    if (data[pos >> 3] >> (pos & 7)) & 1:
                        value |= 1 << k
                yield value

        return cls(_encode_chunks(chunks()), length)

    @classmethod
    def empty(cls, length: int = 0) -> "Bitmap":
        return cls((), length)

    @classmethod
    def full(cls, length: int) -> "Bitmap":
        if length == 0:
            return cls((), 0)
        n_chunks = (length + _CHUNK - 1) // _CHUNK
        valid = length - (n_chunks - 1) * _CHUNK
        last = (1 << valid) - 1 if valid < _CHUNK else _MASK31
        return cls(_encode_chunks([_MASK31] * (n_chunks - 1) + [last]), length)

    @classmethod
    def from_bytes(cls, length: int, data: bytes) -> "Bitmap":
        """Inverse of :meth:`to_bytes` (words packed little-endian u32)."""
        if len(data) % 4:
            raise ValueError("bitmap blob length must be a multiple of 4")
        words = struct.unpack(f"<{len(data) // 4}I", data) if data else ()
        return cls(words, length)

    def to_bytes(self) -> bytes:
        if not self._words:
            return b""
        return struct.pack(f"<{len(self._words)}I", *self._words)

    # -- compressed-form access -------------------------------------------

    @property
    def length(self) -> int:
        return self._length

    @property
    def encoded_size(self) -> int:
        """Bytes occupied by the compressed word stream."""
        return 4 * len(self._words)

    def count(self) -> int:
        """Popcount answered *directly* from the compressed words."""
        total = 0
        for word in self._words:
            if word & _FILL_FLAG:
                if word & _FILL_VAL:
                    total += (word & _COUNT_MASK) * _CHUNK
            else:
                total += (word & _MASK31).bit_count()
        return total

    # -- decoding ----------------------------------------------------------

    def _raw_chunks(self) -> Iterator[int]:
        """Yield exactly ceil(length/31) chunk values (zero-padded)."""
        n_chunks = (self._length + _CHUNK - 1) // _CHUNK
        it = self._iter_words()
        for _ in range(n_chunks):
            yield next(it, 0)

    def _iter_words(self) -> Iterator[int]:
        for word in self._words:
            if word & _FILL_FLAG:
                value = _MASK31 if word & _FILL_VAL else 0
                for _ in range(word & _COUNT_MASK):
                    yield value
            else:
                yield word & _MASK31

    def positions(self) -> list[int]:
        """All set bit positions, in ascending order (materializes rows)."""
        out: list[int] = []
        for index, chunk in enumerate(self._raw_chunks()):
            base = index * _CHUNK
            while chunk:
                low = chunk & -chunk
                bit = low.bit_length() - 1
                pos = base + bit
                if pos < self._length:
                    out.append(pos)
                chunk ^= low
        return out

    def __contains__(self, pos: int) -> bool:
        if not 0 <= pos < self._length:
            return False
        index, bit = divmod(pos, _CHUNK)
        for i, chunk in enumerate(self._raw_chunks()):
            if i == index:
                return bool((chunk >> bit) & 1)
        return False

    def __len__(self) -> int:
        return self._length

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Bitmap):
            return NotImplemented
        return self._length == other._length and list(self._raw_chunks()) == list(
            other._raw_chunks()
        )

    def __repr__(self) -> str:
        return f"<Bitmap length={self._length} count={self.count()} words={len(self._words)}>"

    # -- logical operations -------------------------------------------------

    def _binop(self, other: "Bitmap", fn) -> "Bitmap":
        length = max(self._length, other._length)
        a, b = self._raw_chunks(), other._raw_chunks()
        chunks = [fn(x, y) for x, y in _zip_longest(a, b)]
        return Bitmap(_encode_chunks(chunks), length)

    def and_(self, other: "Bitmap") -> "Bitmap":
        return self._binop(other, lambda x, y: x & y)

    def or_(self, other: "Bitmap") -> "Bitmap":
        return self._binop(other, lambda x, y: x | y)

    def not_(self) -> "Bitmap":
        last = (self._length - 1) // _CHUNK if self._length else -1
        valid = self._length - last * _CHUNK if last >= 0 else 0
        keep = (1 << valid) - 1 if valid < _CHUNK else _MASK31
        chunks = [
            (chunk ^ _MASK31) & (keep if index == last else _MASK31)
            for index, chunk in enumerate(self._raw_chunks())
        ]
        return Bitmap(_encode_chunks(chunks), self._length)


def _zip_longest(a: Iterator[int], b: Iterator[int]) -> Iterator[tuple[int, int]]:
    sentinel = object()
    while True:
        x = next(a, sentinel)
        y = next(b, sentinel)
        if x is sentinel and y is sentinel:
            return
        yield (0 if x is sentinel else x, 0 if y is sentinel else y)


# ---------------------------------------------------------------------------
# Per-column indexes
# ---------------------------------------------------------------------------


class ColumnIndex:
    """Shared behaviour of column indexes (equality + IN + ranges)."""

    kind = "?"

    def __init__(self, name: str, length: int) -> None:
        self.name = name
        self.length = length
        self._eq: dict = {}
        self._empty = Bitmap.empty(length)

    # -- equality ----------------------------------------------------------

    def eq(self, value) -> Bitmap:
        bitmap = self._eq.get(value)
        return self._empty if bitmap is None else bitmap

    def in_(self, values: Iterable) -> Bitmap:
        result = self._empty
        for value in values:
            hit = self._eq.get(value)
            if hit is not None:
                result = result.or_(hit)
        return result

    def ne(self, value) -> Bitmap:
        return self.eq(value).not_()

    # -- ranges (implemented by subclasses) --------------------------------

    def lt(self, value) -> Bitmap:
        raise NotImplementedError

    def le(self, value) -> Bitmap:
        raise NotImplementedError

    def gt(self, value) -> Bitmap:
        raise NotImplementedError

    def ge(self, value) -> Bitmap:
        raise NotImplementedError

    # -- statistics ---------------------------------------------------------

    def iter_bitmaps(self) -> Iterator[Bitmap]:
        yield from self._eq.values()

    @property
    def encoded_size(self) -> int:
        return sum(b.encoded_size for b in self.iter_bitmaps())

    # -- persistence --------------------------------------------------------

    def _meta(self) -> dict:
        raise NotImplementedError

    @classmethod
    def _from_meta(cls, meta: dict, bitmaps: list[Bitmap]) -> "ColumnIndex":
        raise NotImplementedError


class IntColumnIndex(ColumnIndex):
    """Bit-sliced index over non-negative offsets of integer values."""

    kind = "int"

    def __init__(self, name: str, values: Sequence[int]) -> None:
        super().__init__(name, len(values))
        if any(not isinstance(v, int) or isinstance(v, bool) for v in values):
            raise TypeError("IntColumnIndex requires plain int values")
        self.min = min(values) if values else 0
        shifted = [v - self.min for v in values]
        self._ordered = sorted(set(values))
        for value in self._ordered:
            self._eq[value] = Bitmap.from_positions(
                self.length, (i for i, v in enumerate(values) if v == value)
            )
        max_shift = max(shifted) if shifted else 0
        self.nplanes = max(1, max_shift.bit_length())
        self._slices = [
            Bitmap.from_positions(
                self.length, (i for i, s in enumerate(shifted) if (s >> b) & 1)
            )
            for b in range(self.nplanes)
        ]

    # -- bit-slice range algebra --------------------------------------------

    def _lt_shift(self, t: int) -> Bitmap:
        """Rows whose shifted value ``s`` satisfies ``s < t`` (s >= 0)."""
        if t <= 0:
            return self._empty
        if t.bit_length() > self.nplanes:
            return Bitmap.full(self.length)
        bits = [(t >> i) & 1 for i in range(self.nplanes)]
        result = self._empty
        for i in range(self.nplanes):
            if not bits[i]:
                continue
            partial = self._slices[i].not_()
            for j in range(i + 1, self.nplanes):
                slice_j = self._slices[j]
                partial = partial.and_(slice_j if bits[j] else slice_j.not_())
            result = result.or_(partial)
        return result

    def lt(self, value: int) -> Bitmap:
        return self._lt_shift(value - self.min)

    def le(self, value: int) -> Bitmap:
        return self._lt_shift(value + 1 - self.min)

    def gt(self, value: int) -> Bitmap:
        return self.le(value).not_()

    def ge(self, value: int) -> Bitmap:
        return self.lt(value).not_()

    def iter_bitmaps(self) -> Iterator[Bitmap]:
        yield from super().iter_bitmaps()
        yield from self._slices

    def _meta(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "min": self.min,
            "n_slices": len(self._slices),
            "values": self._ordered,
        }

    @classmethod
    def _from_meta(cls, meta: dict, bitmaps: list[Bitmap]) -> "IntColumnIndex":
        values = list(meta["values"])
        # bitmaps arrive as [eq..., slices...]; rebuild without re-scanning rows
        obj = cls.__new__(cls)
        ColumnIndex.__init__(obj, meta["name"], bitmaps[0].length if bitmaps else _len_of(values, meta))
        obj.min = meta["min"]
        obj._ordered = values
        n_eq = len(values)
        obj._eq = dict(zip(values, bitmaps[:n_eq]))
        obj._slices = list(bitmaps[n_eq:])
        obj.nplanes = meta["n_slices"]
        obj._empty = Bitmap.empty(obj.length)
        return obj


def _len_of(values, meta) -> int:
    return int(meta.get("length", 0))


class StrColumnIndex(ColumnIndex):
    """Equality bitmaps over the distinct values of a string column."""

    kind = "str"

    def __init__(self, name: str, values: Sequence[str]) -> None:
        super().__init__(name, len(values))
        if any(not isinstance(v, str) for v in values):
            raise TypeError("StrColumnIndex requires str values")
        self._ordered = sorted(set(values))
        for value in self._ordered:
            self._eq[value] = Bitmap.from_positions(
                self.length, (i for i, v in enumerate(values) if v == value)
            )

    def _combined(self, ordered: Sequence[str]) -> Bitmap:
        result = self._empty
        for value in ordered:
            result = result.or_(self._eq[value])
        return result

    def lt(self, value: str) -> Bitmap:
        i = bisect.bisect_left(self._ordered, value)
        return self._combined(self._ordered[:i])

    def le(self, value: str) -> Bitmap:
        i = bisect.bisect_right(self._ordered, value)
        return self._combined(self._ordered[:i])

    def gt(self, value: str) -> Bitmap:
        i = bisect.bisect_right(self._ordered, value)
        return self._combined(self._ordered[i:])

    def ge(self, value: str) -> Bitmap:
        i = bisect.bisect_left(self._ordered, value)
        return self._combined(self._ordered[i:])

    def _meta(self) -> dict:
        return {"name": self.name, "kind": self.kind, "values": self._ordered}

    @classmethod
    def _from_meta(cls, meta: dict, bitmaps: list[Bitmap]) -> "StrColumnIndex":
        obj = cls.__new__(cls)
        ColumnIndex.__init__(obj, meta["name"], bitmaps[0].length if bitmaps else 0)
        obj._ordered = list(meta["values"])
        obj._eq = dict(zip(obj._ordered, bitmaps))
        obj._empty = Bitmap.empty(obj.length)
        return obj


def _rebuild(meta: dict, bitmaps: list[Bitmap]) -> ColumnIndex:
    if meta["kind"] == "int":
        column = IntColumnIndex._from_meta(meta, bitmaps)
    else:
        column = StrColumnIndex._from_meta(meta, bitmaps)
    if not bitmaps:
        column.length = int(meta.get("length", 0))
        column._empty = Bitmap.empty(column.length)
    return column


# ---------------------------------------------------------------------------
# Query language
# ---------------------------------------------------------------------------

_TWO_CHAR_OPS = ("==", "!=", "<=", ">=")
_ONE_CHAR_OPS = ("=", "<", ">")
_KEYWORDS = {"AND", "OR", "IN", "NOT"}


def _tokenize(text: str) -> list[tuple[str, object, int]]:
    tokens: list[tuple[str, object, int]] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "(),":
            tokens.append((ch, ch, i))
            i += 1
            continue
        if ch in "'\"":
            start = i
            i += 1
            buf: list[str] = []
            while i < n and text[i] != ch:
                if text[i] == "\\" and i + 1 < n:
                    buf.append(text[i + 1])
                    i += 2
                else:
                    buf.append(text[i])
                    i += 1
            if i >= n:
                raise QueryError("unterminated string literal", start)
            tokens.append(("str", "".join(buf), start))
            i += 1
            continue
        if ch.isdigit() or (ch == "-" and i + 1 < n and text[i + 1].isdigit()):
            start = i
            if ch == "-":
                i += 1
            while i < n and text[i].isdigit():
                i += 1
            if i < n and (text[i].isalpha() or text[i] in "_."):
                raise QueryError(f"malformed number near {text[start:i + 1]!r}", start)
            tokens.append(("num", int(text[start:i]), start))
            continue
        if ch.isalpha() or ch == "_":
            start = i
            while i < n and (text[i].isalnum() or text[i] == "_"):
                i += 1
            word = text[start:i]
            tokens.append(("kw" if word.upper() in _KEYWORDS else "ident", word, start))
            continue
        matched = False
        for op in _TWO_CHAR_OPS + _ONE_CHAR_OPS:
            if text.startswith(op, i):
                tokens.append(("op", op, i))
                i += len(op)
                matched = True
                break
        if not matched:
            raise QueryError(f"unexpected character {ch!r}", i)
    tokens.append(("eof", None, n))
    return tokens


class _Parser:
    """Recursive-descent parser: NOT > AND > OR, parentheses allowed."""

    def __init__(self, engine: "BitmapIndex", tokens) -> None:
        self.engine = engine
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos]

    def next(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, kind: str, what: str):
        tok = self.next()
        if tok[0] != kind:
            raise QueryError(f"expected {what}, got {tok[1]!r}", tok[2])
        return tok

    def parse(self) -> Bitmap:
        bitmap = self.parse_or()
        tok = self.peek()
        if tok[0] != "eof":
            raise QueryError(f"unexpected trailing input {tok[1]!r}", tok[2])
        return bitmap

    def parse_or(self) -> Bitmap:
        left = self.parse_and()
        while self._at_keyword("OR"):
            self.next()
            left = left.or_(self.parse_and())
        return left

    def parse_and(self) -> Bitmap:
        left = self.parse_not()
        while self._at_keyword("AND"):
            self.next()
            left = left.and_(self.parse_not())
        return left

    def parse_not(self) -> Bitmap:
        if self._at_keyword("NOT"):
            self.next()
            return self.parse_not().not_()
        return self.parse_primary()

    def _at_keyword(self, word: str) -> bool:
        kind, value, _ = self.peek()
        return kind == "kw" and value.upper() == word

    def parse_primary(self) -> Bitmap:
        kind, value, pos = self.peek()
        if kind == "(":
            self.next()
            inner = self.parse_or()
            self.expect(")", "')'")
            return inner
        if kind != "ident":
            raise QueryError(f"expected a column name, got {value!r}", pos)
        self.next()
        try:
            column = self.engine.column(value)
        except QueryError as exc:
            raise QueryError(str(exc), pos) from None

        not_in = False
        if self._at_keyword("NOT"):
            self.next()
            not_in = True
            kind, value, pos = self.peek()
            if kind != "kw" or value.upper() != "IN":
                raise QueryError("expected IN after NOT", pos)
        if self._at_keyword("IN"):
            self.next()
            self.expect("(", "'(' after IN")
            values = []
            if self.peek()[0] != ")":
                while True:
                    values.append(self._literal(column))
                    kind2, _, _ = self.peek()
                    if kind2 == ",":
                        self.next()
                        continue
                    break
            self.expect(")", "')' closing IN")
            bitmap = column.in_(values)
            return bitmap.not_() if not_in else bitmap

        kind, op, pos = self.next()
        if kind != "op":
            raise QueryError(f"expected a comparison operator, got {op!r}", pos)
        literal = self._literal(column)
        return self._compare(column, op, literal, pos)

    def _literal(self, column: ColumnIndex):
        kind, value, pos = self.next()
        if kind == "num":
            if column.kind == "str":
                return str(value)
            return value
        if kind == "str":
            if column.kind == "int":
                try:
                    return int(value)
                except ValueError:
                    raise QueryError(
                        f"column {column.name!r} is numeric; got {value!r}", pos
                    ) from None
            return value
        if kind == "ident":
            # unquoted word: a friendlier spelling of a string literal
            if column.kind == "int":
                try:
                    return int(value)
                except ValueError:
                    raise QueryError(
                        f"column {column.name!r} is numeric; got {value!r}", pos
                    ) from None
            return value
        raise QueryError(f"expected a literal, got {value!r}", pos)

    @staticmethod
    def _compare(column: ColumnIndex, op: str, literal, pos: int) -> Bitmap:
        if op == "=" or op == "==":
            return column.eq(literal)
        if op == "!=":
            return column.ne(literal)
        if op == "<":
            return column.lt(literal)
        if op == "<=":
            return column.le(literal)
        if op == ">":
            return column.gt(literal)
        if op == ">=":
            return column.ge(literal)
        raise QueryError(f"unsupported operator {op!r}", pos)


class QueryResult:
    """Rows matching a query; rows are materialized only on demand."""

    def __init__(self, bitmap: Bitmap, engine: "BitmapIndex") -> None:
        self._bitmap = bitmap
        self._engine = engine

    @property
    def bitmap(self) -> Bitmap:
        return self._bitmap

    def count(self) -> int:
        """Answered from the compressed bitmap — no row materialization."""
        return self._bitmap.count()

    def __len__(self) -> int:
        return self.count()

    def row_indices(self) -> list[int]:
        return self._bitmap.positions()

    def rows(self) -> list[dict]:
        return [self._engine.rows[i] for i in self._bitmap.positions()]

    def __repr__(self) -> str:
        return f"<QueryResult count={self.count()}>"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _looks_int(values: Sequence[str]) -> bool:
    for v in values:
        try:
            int(v)
        except ValueError:
            return False
    return True


class BitmapIndex:
    """Bit-sliced index over a table of columns; queries never scan rows."""

    def __init__(self) -> None:
        self._columns: dict[str, ColumnIndex] = {}
        self._order: list[str] = []
        self._raw: dict[str, list] = {}
        self._rows: list[dict] = []
        self._length = 0

    # -- table construction -------------------------------------------------

    @property
    def nrows(self) -> int:
        return self._length

    @property
    def columns(self) -> list[str]:
        return list(self._order)

    @property
    def rows(self) -> list[dict]:
        """The underlying rows (only for materialization / verification)."""
        return list(self._rows)

    def column(self, name: str) -> ColumnIndex:
        try:
            return self._columns[name]
        except KeyError:
            raise QueryError(f"unknown column {name!r}") from None

    def add_column(self, name: str, values: Sequence, kind: str | None = None) -> None:
        """Add a column from a Python sequence (ints or strs)."""
        values = list(values)
        if not self._order:
            self._length = len(values)
        elif len(values) != self._length:
            raise ValueError(
                f"column {name!r} has {len(values)} rows, expected {self._length}"
            )
        if kind is None:
            kind = "int" if all(isinstance(v, int) and not isinstance(v, bool) for v in values) else "str"
        if kind == "int":
            column_values = [int(v) for v in values]
            column: ColumnIndex = IntColumnIndex(name, column_values)
        elif kind == "str":
            column_values = [str(v) for v in values]
            column = StrColumnIndex(name, column_values)
        else:
            raise ValueError(f"unknown column kind {kind!r}")
        if name in self._columns:
            self._order.remove(name)
        self._columns[name] = column
        self._order.append(name)
        self._raw[name] = list(column_values)
        self._rebuild_rows()

    def load_csv(
        self, path, *, has_header: bool = True, delimiter: str = ","
    ) -> None:
        """(Re)build the index from a CSV file.

        A column is typed ``int`` when *every* cell parses as an integer,
        otherwise ``str``.
        """
        with open(os.fspath(path), newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            rows = list(reader)
        self._columns.clear()
        self._order.clear()
        self._raw = {}
        self._rows = []
        self._length = 0
        if not rows:
            return
        if has_header:
            header, body = rows[0], rows[1:]
        else:
            width = len(rows[0])
            header = [f"col{i}" for i in range(width)]
            body = rows
        width = len(header)
        body = [r for r in body if r]
        for r in body:
            if len(r) != width:
                raise ValueError(
                    f"CSV row has {len(r)} fields, expected {width}: {r!r}"
                )
        self._length = len(body)
        for i, name in enumerate(header):
            cells = [r[i] for r in body]
            kind = "int" if cells and _looks_int(cells) else "str"
            if kind == "int":
                values = [int(c) for c in cells]
                self._columns[name] = IntColumnIndex(name, values)
            else:
                values = list(cells)
                self._columns[name] = StrColumnIndex(name, values)
            self._order.append(name)
            self._raw[name] = values
        self._rebuild_rows()

    def _rebuild_rows(self) -> None:
        """Materialize row dicts from the stored raw column values."""
        if not self._order or self._length == 0:
            self._rows = []
            return
        columns = [self._raw[name] for name in self._order]
        order = self._order
        self._rows = [
            {order[j]: columns[j][i] for j in range(len(order))}
            for i in range(self._length)
        ]

    # -- querying -----------------------------------------------------------

    def query(self, expr: str) -> QueryResult:
        """Parse ``expr`` and return matching rows as a lazy bitmap result."""
        if not isinstance(expr, str):
            raise TypeError("query expression must be a string")
        tokens = _tokenize(expr)
        bitmap = _Parser(self, tokens).parse()
        return QueryResult(bitmap, self)

    def count(self, expr: str) -> int:
        """``COUNT(expr)`` answered straight from the compressed bitmap."""
        return self.query(expr).count()

    # -- persistence --------------------------------------------------------

    def save(self, path) -> None:
        """Serialize the index to ``path`` (deterministic binary format)."""
        with open(os.fspath(path), "wb") as fh:
            fh.write(_MAGIC)
            fh.write(struct.pack("<III", self._length, len(self._order), len(self._rows)))
            for name in self._order:
                column = self._columns[name]
                meta = column._meta()
                meta["length"] = column.length
                blob = json.dumps(meta, sort_keys=True).encode("utf-8")
                bitmaps = [column._eq[v] for v in meta["values"]]
                if column.kind == "int":
                    bitmaps += column._slices
                fh.write(struct.pack("<I", len(blob)))
                fh.write(blob)
                fh.write(struct.pack("<I", len(bitmaps)))
                for bitmap in bitmaps:
                    data = bitmap.to_bytes()
                    fh.write(struct.pack("<I", len(data)))
                    fh.write(data)
            # raw rows so that materialization survives a reload
            rows_blob = json.dumps(self._rows, sort_keys=True).encode("utf-8")
            fh.write(struct.pack("<I", len(rows_blob)))
            fh.write(rows_blob)

    @classmethod
    def load(cls, path) -> "BitmapIndex":
        """Load an index previously written by :meth:`save`."""
        engine = cls()
        with open(os.fspath(path), "rb") as fh:
            magic = fh.read(len(_MAGIC))
            if magic != _MAGIC:
                raise ValueError("not a bitmap-index file (bad magic)")
            length, ncols, nrows = struct.unpack("<III", fh.read(12))
            engine._length = length
            for _ in range(ncols):
                (meta_len,) = struct.unpack("<I", fh.read(4))
                meta = json.loads(fh.read(meta_len).decode("utf-8"))
                (n_bitmaps,) = struct.unpack("<I", fh.read(4))
                bitmaps = []
                for _ in range(n_bitmaps):
                    (data_len,) = struct.unpack("<I", fh.read(4))
                    bitmaps.append(Bitmap.from_bytes(length, fh.read(data_len)))
                column = _rebuild(meta, bitmaps)
                column.length = length
                engine._columns[column.name] = column
                engine._order.append(column.name)
            (rows_len,) = struct.unpack("<I", fh.read(4))
            engine._rows = json.loads(fh.read(rows_len).decode("utf-8"))
        if len(engine._rows) != nrows:
            raise ValueError("corrupt index: row count mismatch")
        engine._raw = {
            name: [row[name] for row in engine._rows] for name in engine._order
        }
        return engine

    def __repr__(self) -> str:
        return (
            f"<BitmapIndex rows={self._length} columns={len(self._order)}>"
        )
