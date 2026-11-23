# Copyright (c) 2026 EPAM Systems
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Executes the repository/branch back-fill against a real Postgres (#6962).

The back-fill fails softly, so a statement that never runs would leave old rows without
provenance while every run still reports success. Skipped without a reachable Postgres.
"""

import json

import sqlalchemy as sa

from elitea_sdk.tools.vector_adapters.VectorStoreAdapter import IDENTITY_STAMP_BATCH_SIZE, PGVectorAdapter
from tests.tools.index.test_6645_identity_stamp_live_sql import _seed, _wrapper_for, embedding_table  # noqa: F401


def _metadata(engine, schema):
    with engine.connect() as connection:
        return dict(connection.execute(sa.text(
            f'SELECT id, cmetadata FROM "{schema}".langchain_pg_embedding'
        )).all())


class TestTheTargetReachesTheRows:
    def test_rows_gain_repository_and_branch_and_keep_their_metadata(self, embedding_table):
        schema, engine = embedding_table
        _seed(engine, schema, [("row-1", '{"filename": "a.py", "commit_hash": "h1", "branch": "old"}'),
                               ("row-2", None)])

        stamped = PGVectorAdapter().stamp_source_target(
            _wrapper_for(schema, engine), ["row-1", "row-2"], {"repository": "org/repo", "branch": "main"})

        assert stamped == 2
        assert _metadata(engine, schema) == {
            "row-1": {"filename": "a.py", "commit_hash": "h1", "repository": "org/repo", "branch": "main"},
            "row-2": {"repository": "org/repo", "branch": "main"},
        }

    def test_rows_outside_the_list_are_untouched(self, embedding_table):
        schema, engine = embedding_table
        _seed(engine, schema, [("row-1", '{"filename": "a.py"}'), ("row-2", '{"filename": "b.py"}')])

        PGVectorAdapter().stamp_source_target(_wrapper_for(schema, engine), ["row-1"], {"repository": "org/repo"})

        assert _metadata(engine, schema)["row-2"] == {"filename": "b.py"}

    def test_more_rows_than_one_batch(self, embedding_table):
        schema, engine = embedding_table
        row_ids = [f"row-{n}" for n in range(IDENTITY_STAMP_BATCH_SIZE + 3)]
        _seed(engine, schema, [(row_id, json.dumps({"n": row_id})) for row_id in row_ids])

        stamped = PGVectorAdapter().stamp_source_target(_wrapper_for(schema, engine), row_ids, {"branch": "dev"})

        assert stamped == len(row_ids)
        assert all(meta["branch"] == "dev" for meta in _metadata(engine, schema).values())
