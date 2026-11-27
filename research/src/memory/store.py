"""Protocol migration on the project's existing SQLite connection and owner guard.

The legacy event/food tables and their hashes remain untouched. The new protocol
needs immutable structured scopes and substitution edges; migration 1 adds only
namespaced tables instead of reinterpreting old pilot data.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from memory.store import Store as LegacyStore


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Store(LegacyStore):
    def __init__(self, path, person_id="research-user"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        super().__init__(path, person_id)
        self.path = str(path)
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS cm_migrations(version INTEGER PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS cm_records(
          seq INTEGER PRIMARY KEY AUTOINCREMENT,
          memory_id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS cm_executions(
          seq INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL);
        INSERT OR IGNORE INTO cm_migrations VALUES (1);
        """)
        self.conn.commit()

    def initialize(self, records):
        if self.list_records():
            if canonical(self.list_records()) != canonical(records):
                raise ValueError("INITIAL_MEMORY_ALREADY_EXISTS")
            return
        from .executor import validate_record
        with self.conn:
            for record in records:
                validate_record(record, initial=True)
                self.conn.execute("INSERT INTO cm_records(memory_id,payload) VALUES (?,?)",
                                  (record["memory_id"], canonical(record)))

    def list_records(self):
        return [json.loads(row[0]) for row in self.conn.execute(
            "SELECT payload FROM cm_records ORDER BY seq")]

    def snapshot(self):
        return deepcopy(self.list_records())

    def hash(self):
        return hashlib.sha256(canonical(self.list_records()).encode()).hexdigest()

    def next_id(self):
        occupied = {r["memory_id"] for r in self.list_records()}
        index = len(occupied) + 1
        while f"m{index}" in occupied:
            index += 1
        return f"m{index}"

    def close(self):
        self.conn.close()


MemoryStore = Store
