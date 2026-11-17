# Bitfold — Compressed Bitmap Index for Analytics

Bit-sliced indexing over CSV columns that answers boolean analytics queries
**without materializing a single row**. Build once, then `= / IN / < <= > >= /
AND / OR / NOT` queries run as bitwise algebra over WAH-compressed bitmaps, and
`COUNT` is computed straight from the compressed words.

**Why it's special:** this is the storage technique behind real columnar
databases (Oracle Bitmap indexes, Druid/Roaring-style engines), implemented
from scratch in one dependency-free module — including the bit-slice range
trick, so `age < 45` costs a few bitmap operations instead of scanning 300
rows.

## Architecture

```
CSV ──▶ type inference (int | str)
        ├─ str column:  equality bitmap per distinct value (+ bisect for ranges)
        └─ int column:  equality bitmaps + bit-slices over (value - min)
                             └─ "col < x" via the classic identity:
                                s < t  =  OR over set bits i of t:
                                          (AND over j>i: s_j == t_j) AND s_i == 0

Bitmap storage — WAH-style chunked run-length:
  bit stream cut into 31-bit chunks →
    literal word (flag=0, 31 bits)  |  fill word (flag=1, value, repeat count)
  runs of uniform chunks collapse to one word; count() walks these words
  directly (popcount of literals + count×31 of fills) — rows never enumerated.

Query language:  =  ==  !=  <  <=  >  >=  IN (...)  NOT IN (...)  AND  OR  NOT
with parentheses; NOT > AND > OR precedence.
```

- **`Bitmap`** — immutable compressed bit vector: `count()` (compressed-form),
  `positions()`, `and_/or_/not_`, `to_bytes/from_bytes`.
- **`IntColumnIndex` / `StrColumnIndex`** — per-column equality + range
  bitmaps.
- **`BitmapIndex`** — `load_csv` / `add_column`, `query(expr)`,
  `count(expr)`, `save(path)` / `load(path)` (deterministic binary format
  with `BIDX1` magic).
- **`QueryResult`** — lazy: `count()` from the bitmap; `rows()` materializes
  only on demand.

## Usage example (runnable)

```python
import os, random, tempfile
from bitmap_index import BitmapIndex

rnd = random.Random(4)
eng = BitmapIndex()
eng.add_column("city", [rnd.choice(["Berlin", "Oslo", "Rome"]) for _ in range(6)])
eng.add_column("age", [rnd.randint(20, 70) for _ in range(6)])

print("rows:", eng.rows)
print("city='Rome'      ->", eng.query("city = 'Rome'").row_indices())
print("age >= 50        ->", eng.query("age >= 50").row_indices())
print("combined COUNT   ->", eng.count("city = 'Oslo' AND age >= 40"))

# persistence
path = os.path.join(tempfile.gettempdir(), "demo.bidx")
eng.save(path)
again = BitmapIndex.load(path)
print("after save/load  ->", again.count("city = 'Oslo' AND age >= 40"))
os.remove(path)
```

Output:

```
rows: [{'city': 'Berlin', 'age': 29}, {'city': 'Oslo', 'age': 25}, {'city': 'Berlin', 'age': 24}, {'city': 'Rome', 'age': 21}, {'city': 'Oslo', 'age': 45}, {'city': 'Oslo', 'age': 55}]
city='Rome'      -> [3]
age >= 50        -> [5]
combined COUNT   -> 2
after save/load  -> 2
```

From a file: `eng = BitmapIndex(); eng.load_csv("sales.csv")` — a column is
typed `int` only when *every* cell parses as an integer.

## Running tests

From this directory:

```
python -m unittest discover -s tests -v
```

35 tests: every query type cross-checked against brute-force row scans on a
seeded random fixture (including negative integers and string ranges),
measurable compression of sorted columns, count-without-materialization
(asserted by patching `positions` to explode), persistence roundtrip,
empty/full-result edges, CSV type inference, and query errors with positions.

## License

MIT — see `LICENSE`.
