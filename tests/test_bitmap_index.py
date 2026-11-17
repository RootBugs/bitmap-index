"""Tests for the bitmap-index bit-sliced index library."""

import csv
import os
import random
import shutil
import tempfile
import unittest
from unittest import mock

import bitmap_index
from bitmap_index import (
    Bitmap,
    BitmapIndex,
    IntColumnIndex,
    QueryError,
    StrColumnIndex,
)

CITIES = ["Berlin", "Lisbon", "Oslo", "Paris", "Quito", "Rome", "Sofia"]


def build_engine(n=300, seed=1234):
    rnd = random.Random(seed)
    ages = [rnd.randint(18, 80) for _ in range(n)]
    salaries = [rnd.randint(-50, 300) for _ in range(n)]
    cities = [rnd.choice(CITIES) for _ in range(n)]
    grades = [rnd.choice("ABCD") for _ in range(n)]
    eng = BitmapIndex()
    eng.add_column("id", list(range(n)))
    eng.add_column("age", ages)
    eng.add_column("salary", salaries)
    eng.add_column("city", cities)
    eng.add_column("grade", grades)
    return eng


class BitmapUnitTests(unittest.TestCase):
    def test_from_positions_count_and_positions(self):
        bm = Bitmap.from_positions(70, [0, 1, 31, 32, 63, 69])
        self.assertEqual(bm.count(), 6)
        self.assertEqual(bm.positions(), [0, 1, 31, 32, 63, 69])
        self.assertEqual(bm.length, 70)
        self.assertIn(31, bm)
        self.assertNotIn(70, bm)
        self.assertNotIn(64, bm)

    def test_count_comes_from_compressed_words_not_rows(self):
        # 10_000 set bits in one run compresses to a couple of fill words
        bm = Bitmap.from_positions(10000, range(5000, 10000))
        self.assertLessEqual(bm.encoded_size, 24)
        self.assertEqual(bm.count(), 5000)

    def test_logical_ops_and_not(self):
        a = Bitmap.from_positions(10, [1, 2, 3, 4])
        b = Bitmap.from_positions(10, [3, 4, 5, 6])
        self.assertEqual(a.and_(b).positions(), [3, 4])
        self.assertEqual(a.or_(b).positions(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(a.not_().positions(), [0, 5, 6, 7, 8, 9])
        self.assertEqual(a.and_(b).count(), 2)
        # NOT is an involution and De Morgan holds
        self.assertEqual(a.not_().not_(), a)
        self.assertEqual(a.and_(b).not_(), a.not_().or_(b.not_()))

    def test_not_masks_bits_beyond_logical_length(self):
        # length 10: NOT must not count the 21 padding bits of the chunk
        bm = Bitmap.full(10)
        self.assertEqual(bm.count(), 10)
        self.assertEqual(bm.not_().count(), 0)
        self.assertEqual(Bitmap.empty(10).not_().positions(), list(range(10)))

    def test_binop_pads_shorter_bitmap(self):
        a = Bitmap.from_positions(5, [0, 1])
        b = Bitmap.from_positions(40, [39])
        merged = a.or_(b)
        self.assertEqual(merged.length, 40)
        self.assertEqual(merged.positions(), [0, 1, 39])
        self.assertEqual(a.and_(b).count(), 0)

    def test_bytes_roundtrip_and_validation(self):
        bm = Bitmap.from_positions(96, [0, 5, 63, 64, 95])
        again = Bitmap.from_bytes(bm.length, bm.to_bytes())
        self.assertEqual(again, bm)
        self.assertEqual(again.positions(), bm.positions())
        with self.assertRaises(ValueError):
            Bitmap.from_bytes(10, b"abc")
        with self.assertRaises(ValueError):
            Bitmap.from_positions(10, [42])
        with self.assertRaises(ValueError):
            Bitmap.from_positions(-1, [])

    def test_empty_bitmap(self):
        empty = Bitmap.empty(0)
        self.assertEqual(empty.count(), 0)
        self.assertEqual(empty.positions(), [])
        self.assertEqual(empty.not_().count(), 0)


class QueryVsBruteForceTests(unittest.TestCase):
    """Every query type must match a plain row scan on a random fixture."""

    @classmethod
    def setUpClass(cls):
        cls.eng = build_engine()
        cls.rows = cls.eng.rows

    def assert_matches(self, expr, predicate):
        expected = [i for i, row in enumerate(self.rows) if predicate(row)]
        self.assertEqual(self.eng.query(expr).row_indices(), expected, expr)
        self.assertEqual(self.eng.count(expr), len(expected), expr)

    def test_equality_matches_brute_force(self):
        for city in CITIES + ["Atlantis"]:
            self.assert_matches(
                f"city = '{city}'", lambda row, c=city: row["city"] == c
            )
        self.assert_matches("grade = 'B'", lambda row: row["grade"] == "B")
        # == and = are synonyms; != is the complement
        self.assert_matches("age == 45", lambda row: row["age"] == 45)
        self.assert_matches("age != 45", lambda row: row["age"] != 45)

    def test_in_query_matches_brute_force(self):
        subset = ["Oslo", "Rome", "Quito"]
        quoted = ", ".join(f"'{c}'" for c in subset)
        self.assert_matches(
            f"city IN ({quoted})", lambda row: row["city"] in subset
        )
        self.assert_matches(
            f"city NOT IN ({quoted})", lambda row: row["city"] not in subset
        )
        self.assert_matches("city IN ('Atlantis')", lambda row: False)

    def test_int_range_queries_match_brute_force(self):
        for threshold in (0, 17, 18, 45, 60, 80, 81, 1000):
            self.assert_matches(
                f"age < {threshold}", lambda row, t=threshold: row["age"] < t
            )
            self.assert_matches(
                f"age <= {threshold}", lambda row, t=threshold: row["age"] <= t
            )
            self.assert_matches(
                f"age > {threshold}", lambda row, t=threshold: row["age"] > t
            )
            self.assert_matches(
                f"age >= {threshold}", lambda row, t=threshold: row["age"] >= t
            )

    def test_negative_int_ranges_match_brute_force(self):
        # salary includes negatives; min is below zero so the bit-slice
        # offset arithmetic must hold for negative thresholds too
        observed_min = min(row["salary"] for row in self.rows)
        observed_max = max(row["salary"] for row in self.rows)
        for threshold in (observed_min - 1, observed_min, -1, 0, 50,
                          observed_max, observed_max + 1):
            self.assert_matches(
                f"salary < {threshold}",
                lambda row, t=threshold: row["salary"] < t,
            )
            self.assert_matches(
                f"salary >= {threshold}",
                lambda row, t=threshold: row["salary"] >= t,
            )

    def test_string_range_queries_match_brute_force(self):
        for threshold in ("A", "Berlin", "P", "Rome", "Zz"):
            self.assert_matches(
                f"city < '{threshold}'",
                lambda row, t=threshold: row["city"] < t,
            )
            self.assert_matches(
                f"city >= '{threshold}'",
                lambda row, t=threshold: row["city"] >= t,
            )

    def test_boolean_composition_matches_brute_force(self):
        self.assert_matches(
            "age >= 30 AND city = 'Oslo'",
            lambda row: row["age"] >= 30 and row["city"] == "Oslo",
        )
        self.assert_matches(
            "(age < 25 OR age > 75) AND grade != 'D'",
            lambda row: (row["age"] < 25 or row["age"] > 75) and row["grade"] != "D",
        )
        self.assert_matches(
            "NOT city = 'Rome' AND salary < 0 OR grade = 'A'",
            # precedence: (NOT city='Rome' AND salary<0) OR grade='A'
            lambda row: (
                (row["city"] != "Rome" and row["salary"] < 0)
                or row["grade"] == "A"
            ),
        )
        self.assert_matches(
            "NOT (grade = 'A' OR grade = 'B')",
            lambda row: row["grade"] not in ("A", "B"),
        )

    def test_and_binds_tighter_than_or(self):
        # age > 70 OR (age < 20 AND grade = 'A') — not (age > 70 OR age < 20) AND A
        self.assert_matches(
            "age > 70 OR age < 20 AND grade = 'A'",
            lambda row: row["age"] > 70 or (row["age"] < 20 and row["grade"] == "A"),
        )

    def test_count_never_materializes_rows(self):
        expected = sum(1 for row in self.rows if row["age"] >= 30)
        with mock.patch.object(
            Bitmap, "positions", side_effect=AssertionError("rows materialized!")
        ):
            self.assertEqual(self.eng.count("age >= 30"), expected)
            # a COUNT over the compressed bitmap must equal the row count
            self.assertEqual(Bitmap.from_positions(5000, range(5000)).count(), 5000)
        # outside the patch, indices agree with count
        result = self.eng.query("age >= 30")
        self.assertEqual(result.count(), len(result.row_indices()))

    def test_empty_and_full_result_edges(self):
        self.assertEqual(self.eng.count("city = 'Atlantis'"), 0)
        self.assertEqual(self.eng.query("city = 'Atlantis'").row_indices(), [])
        self.assertEqual(self.eng.count("salary > 100000"), 0)
        self.assertEqual(self.eng.count("city > 'A'"), len(self.rows))
        self.assertEqual(self.eng.count("age >= 0"), len(self.rows))
        self.assertEqual(
            self.eng.query("age >= 0").row_indices(), list(range(len(self.rows)))
        )
        self.assertEqual(self.eng.count("city IN ()"), 0)


class CompressionTests(unittest.TestCase):
    def test_compression_shrinks_a_sorted_column_measurably(self):
        grouped = ["alpha"] * 400 + ["beta"] * 400 + ["gamma"] * 200
        shuffled = list(grouped)
        random.Random(7).shuffle(shuffled)

        sorted_eng = BitmapIndex()
        sorted_eng.add_column("c", grouped)
        shuffled_eng = BitmapIndex()
        shuffled_eng.add_column("c", shuffled)

        sorted_size = sorted_eng.column("c").encoded_size
        shuffled_size = shuffled_eng.column("c").encoded_size
        naive_size = 3 * 1000 // 8  # three uncompressed bitmaps of 1000 bits

        # sorted column compresses at least 4x vs the naive bitset,
        # and clearly beats the shuffled arrangement of the same values
        self.assertLess(sorted_size * 4, naive_size)
        self.assertLess(sorted_size * 4, shuffled_size)
        # both arrangements answer identically
        self.assertEqual(sorted_eng.count("c = 'beta'"), 400)
        self.assertEqual(shuffled_eng.count("c = 'beta'"), 400)

    def test_sorted_int_column_equality_bitmaps_are_compact(self):
        values = [i // 100 for i in range(1000)]  # 100-long runs of each value
        eng = BitmapIndex()
        eng.add_column("v", values)
        column = eng.column("v")
        # 10 uncompressed bitmaps would be 10 * 125 bytes; run-length
        # encoding must beat that by a wide margin
        naive = 10 * (1000 // 8)
        self.assertLess(column.encoded_size * 2, naive)
        self.assertEqual(eng.count("v = 5"), 100)
        self.assertEqual(eng.count("v >= 9"), 100)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="bidx-test-")
        self.path = os.path.join(self.dir, "index.bidx")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_save_load_roundtrip_preserves_every_query(self):
        eng = build_engine()
        eng.save(self.path)
        with open(self.path, "rb") as fh:
            self.assertEqual(fh.read(5), b"BIDX1")
        again = BitmapIndex.load(self.path)
        self.assertEqual(again.columns, eng.columns)
        self.assertEqual(again.nrows, eng.nrows)
        self.assertEqual(again.rows, eng.rows)
        for expr in (
            "city = 'Rome'",
            "age >= 30 AND salary < 100",
            "city IN ('Oslo', 'Paris') OR NOT grade = 'A'",
            "age < 20 OR age > 75",
            "salary <= -40",
        ):
            self.assertEqual(
                again.query(expr).row_indices(), eng.query(expr).row_indices(), expr
            )
            self.assertEqual(again.count(expr), eng.count(expr), expr)

    def test_load_rejects_corrupt_magic(self):
        eng = build_engine(n=5)
        eng.save(self.path)
        with open(self.path, "r+b") as fh:
            fh.seek(0)
            fh.write(b"NOPE!")
        with self.assertRaises(ValueError):
            BitmapIndex.load(self.path)

    def test_loaded_engine_can_be_compounded_with_rows(self):
        build_engine(n=10).save(self.path)
        again = BitmapIndex.load(self.path)
        self.assertEqual(len(again.rows), 10)
        self.assertEqual(again.rows[0], {"id": 0, "age": again.rows[0]["age"],
                                         "salary": again.rows[0]["salary"],
                                         "city": again.rows[0]["city"],
                                         "grade": again.rows[0]["grade"]})


class CsvLoadingTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="bidx-csv-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write(self, name, rows):
        path = os.path.join(self.dir, name)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(rows)
        return path

    def test_load_csv_builds_indexes_matching_file_content(self):
        path = self._write("t.csv", [
            ["id", "age", "city"],
            ["0", "25", "Rome"],
            ["1", "40", "Paris"],
            ["2", "40", "Rome"],
            ["3", "-3", "Oslo"],
        ])
        eng = BitmapIndex()
        eng.load_csv(path)
        self.assertEqual(eng.columns, ["id", "age", "city"])
        self.assertEqual(eng.nrows, 4)
        # int inference includes negatives; age is int, city is str
        self.assertEqual(eng.column("age").kind, "int")
        self.assertEqual(eng.column("city").kind, "str")
        self.assertEqual(eng.rows[3], {"id": 3, "age": -3, "city": "Oslo"})
        self.assertEqual(eng.query("city = 'Rome'").row_indices(), [0, 2])
        self.assertEqual(eng.count("age >= 40"), 2)
        self.assertEqual(eng.count("age < 0"), 1)

    def test_header_only_csv_yields_empty_engine(self):
        path = self._write("empty.csv", [["id", "name"]])
        eng = BitmapIndex()
        eng.load_csv(path)
        self.assertEqual(eng.nrows, 0)
        self.assertEqual(eng.columns, ["id", "name"])
        self.assertEqual(eng.count("id >= 0"), 0)
        self.assertEqual(eng.query("name = 'x'").row_indices(), [])

    def test_totally_empty_file_yields_no_columns(self):
        path = self._write("nothing.csv", [])
        eng = BitmapIndex()
        eng.load_csv(path)
        self.assertEqual(eng.nrows, 0)
        self.assertEqual(eng.columns, [])
        with self.assertRaises(QueryError):
            eng.query("ghost = 1")

    def test_ragged_csv_row_raises_with_field_count(self):
        path = self._write("bad.csv", [["a", "b"], ["1", "2", "3"]])
        eng = BitmapIndex()
        with self.assertRaises(ValueError) as ctx:
            eng.load_csv(path)
        self.assertIn("expected 2", str(ctx.exception))

    def test_add_column_rejects_length_mismatch(self):
        eng = BitmapIndex()
        eng.add_column("a", [1, 2, 3])
        with self.assertRaises(ValueError):
            eng.add_column("b", [1, 2])


class QueryErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = build_engine(n=50)

    def test_unknown_column_reports_position(self):
        with self.assertRaises(QueryError) as ctx:
            self.eng.query("nonsense > 5")
        self.assertIn("unknown column", str(ctx.exception))
        self.assertIn("position 0", str(ctx.exception))

    def test_syntax_error_reports_position(self):
        with self.assertRaises(QueryError) as ctx:
            self.eng.query("age >= ")
        self.assertIn("position", str(ctx.exception))
        with self.assertRaises(QueryError):
            self.eng.query("age >< 5")

    def test_unterminated_string_rejected(self):
        with self.assertRaises(QueryError) as ctx:
            self.eng.query("city = 'Rome")
        self.assertIn("unterminated", str(ctx.exception))

    def test_wrong_literal_type_rejected(self):
        with self.assertRaises(QueryError) as ctx:
            self.eng.query("age >= 'banana'")
        self.assertIn("numeric", str(ctx.exception))

    def test_malformed_tokens_rejected(self):
        with self.assertRaises(QueryError):
            self.eng.query("age 12abc > 3")
        with self.assertRaises(QueryError):
            self.eng.query("age $ 3")
        with self.assertRaises(QueryError):
            self.eng.query("age IN 5")
        with self.assertRaises(QueryError):
            self.eng.query("age NOT BETWEEN 1 AND 2")
        with self.assertRaises(TypeError):
            self.eng.query(42)


class ColumnIndexTests(unittest.TestCase):
    def test_int_column_requires_plain_ints(self):
        with self.assertRaises(TypeError):
            IntColumnIndex("x", ["1", "2"])
        with self.assertRaises(TypeError):
            IntColumnIndex("x", [True, False])

    def test_str_column_requires_strings(self):
        with self.assertRaises(TypeError):
            StrColumnIndex("x", [1, 2])

    def test_unknown_column_lookup_raises_query_error(self):
        eng = BitmapIndex()
        eng.add_column("a", [1, 2])
        with self.assertRaises(QueryError):
            eng.column("b")

    def test_index_repr_and_result_len(self):
        eng = BitmapIndex()
        eng.add_column("a", [1, 2, 3])
        self.assertEqual(eng.count("a >= 2"), 2)
        self.assertEqual(len(eng.query("a >= 2")), 2)
        self.assertIn("rows=3", repr(eng))
        self.assertIn("count=2", repr(eng.query("a >= 2")))


if __name__ == "__main__":
    unittest.main()
