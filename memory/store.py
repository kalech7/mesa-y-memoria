import hashlib
import json
import sqlite3
from dataclasses import asdict


class Store:
    def __init__(self, path, person_id):
        self.person_id = person_id
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS owner (person_id TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS memories (
          event_id TEXT PRIMARY KEY, at TEXT NOT NULL, session_id TEXT NOT NULL,
          role TEXT NOT NULL, channel TEXT NOT NULL, source TEXT NOT NULL,
          text TEXT NOT NULL, raw_text TEXT NOT NULL, origin_id TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'active', linked_ids TEXT NOT NULL DEFAULT '[]'
        );
        CREATE TABLE IF NOT EXISTS operations (
          seq INTEGER PRIMARY KEY, op TEXT, event_id TEXT, at TEXT, details TEXT
        );
        CREATE TABLE IF NOT EXISTS food_preferences (
          preference_id TEXT PRIMARY KEY, evidence_id TEXT NOT NULL,
          payload TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS food_restrictions (
          restriction_id TEXT PRIMARY KEY, evidence_id TEXT NOT NULL,
          payload TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS event_metadata (
          event_id TEXT PRIMARY KEY, payload TEXT NOT NULL
        );
        """)
        owners = [r[0] for r in self.conn.execute("SELECT person_id FROM owner")]
        if owners and owners != [person_id]:
            self.conn.close()
            raise ValueError("Base de memoria pertenece a otra persona")
        self.conn.execute("INSERT OR IGNORE INTO owner VALUES (?)", (person_id,))
        self.conn.commit()

    def add(self, message, text=None, event_id=None, source=None, linked_ids=()):
        event_id = event_id or message["message_id"]
        metadata = {k: message[k] for k in ("context", "valid_from", "valid_to", "supersedes",
                                           "food_aliases", "meal_aliases", "confidence") if k in message}
        values = (event_id, message["at"], message["session_id"], message["role"], message["channel"],
                  source or message["source"], text if text is not None else message["text"],
                  message["text"], message["message_id"], "active", json.dumps(list(linked_ids)))
        old = self.conn.execute("SELECT * FROM memories WHERE event_id=?", (event_id,)).fetchone()
        if old:
            if tuple(old)[:9] != values[:9] or old["linked_ids"] != values[-1]:
                raise ValueError("Colisión no idempotente de memoria")
            previous = self.conn.execute("SELECT payload FROM event_metadata WHERE event_id=?", (event_id,)).fetchone()
            if (json.loads(previous[0]) if previous else {}) != metadata:
                raise ValueError("Colisión no idempotente de contexto/vigencia")
            return event_id
        with self.conn:
            self.conn.execute("INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?)", values)
            self.conn.execute("INSERT INTO operations(op,event_id,at,details) VALUES ('ADD',?,?,?)",
                              (event_id, message["at"], json.dumps({"links": list(linked_ids)})))
            if metadata:
                self.conn.execute("INSERT INTO event_metadata VALUES (?,?)",
                                  (event_id, json.dumps(metadata, ensure_ascii=False, sort_keys=True)))
        return event_id

    def _add_food_record(self, record, table, id_field):
        """Append immutable extraction backed by a stored event; never overwrite tastes."""
        payload = asdict(record)
        if payload["user_id"] != self.person_id:
            raise ValueError("Registro alimentario pertenece a otra persona")
        origin = self.conn.execute("SELECT 1 FROM memories WHERE origin_id=?",
                                   (payload["evidence_id"],)).fetchone()
        if not origin:
            raise ValueError("Preferencia/restricción sin evidencia persistida")
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        identifier = payload[id_field]
        old = self.conn.execute(f"SELECT payload FROM {table} WHERE {id_field}=?", (identifier,)).fetchone()
        if old:
            if old[0] != serialized:
                raise ValueError("Colisión no idempotente de extracción alimentaria")
            return identifier
        with self.conn:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?)",
                              (identifier, payload["evidence_id"], serialized))
            self.conn.execute("INSERT INTO operations(op,event_id,at,details) VALUES (?,?,?,?)",
                              ("EXTRACT_PREFERENCE" if table == "food_preferences" else "EXTRACT_RESTRICTION",
                               payload["evidence_id"], payload["timestamp"], serialized))
        return identifier

    def add_preference(self, preference):
        return self._add_food_record(preference, "food_preferences", "preference_id")

    def add_restriction(self, restriction):
        return self._add_food_record(restriction, "food_restrictions", "restriction_id")

    def preferences(self):
        from simulator.preferences import Preference
        return self._food_records("food_preferences", "preference_id", Preference)

    def restrictions(self):
        from simulator.preferences import Restriction
        return self._food_records("food_restrictions", "restriction_id", Restriction)

    def _food_records(self, table, id_field, cls):
        records = []
        for row in self.conn.execute(f"SELECT payload FROM {table} ORDER BY {id_field}"):
            payload = json.loads(row[0])
            for field in ("weekdays", "supersedes", "scope_unknown"):
                if field in payload:
                    payload[field] = tuple(payload[field])
            records.append(cls(**payload))
        return records

    def food_snapshot(self):
        """Includes derived data and explicit event metadata without changing legacy hash."""
        value = {"events": self.all(), "preferences": [asdict(p) for p in self.preferences()],
                 "restrictions": [asdict(r) for r in self.restrictions()],
                 "metadata": [dict(r) for r in self.conn.execute("SELECT * FROM event_metadata ORDER BY event_id")]}
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def clarify(self, message, previous_ids, text=None, resolved=False):
        if not resolved or message["role"] != "user" or message["source"] != "clarification":
            raise ValueError("No persistir una corrección no comunicada/resuelta")
        previous_ids = list(previous_ids)
        existing = {m["event_id"] for m in self.all()}
        if not set(previous_ids) <= existing:
            raise ValueError("Vínculo a memoria inexistente")
        event = self.add(message, text=text, linked_ids=previous_ids)
        with self.conn:
            for previous in previous_ids:
                self.conn.execute("UPDATE memories SET status='superseded' WHERE event_id=?", (previous,))
                self.conn.execute("INSERT INTO operations(op,event_id,at,details) VALUES ('INACTIVATE',?,?,?)",
                    (previous, message["at"], json.dumps({"superseded_by": event})))
        return event

    def all(self):
        return [dict(r) for r in self.conn.execute("SELECT * FROM memories ORDER BY at,event_id")]

    def snapshot(self):
        return hashlib.sha256(json.dumps(self.all(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def backup(self, path):
        # Nunca reemplazar snapshots de otra ejecución.
        if path.exists():
            raise FileExistsError(path)
        target = sqlite3.connect(str(path))
        try:
            self.conn.backup(target)
        finally:
            target.close()

    def close(self):
        self.conn.close()

