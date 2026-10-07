"""Read database identity and migration compatibility without changing the database."""

import ast
import json
import os
import sys
from pathlib import Path

import psycopg2

expected = sys.argv[1]
connection = psycopg2.connect(
    host=os.environ['POSTGRES_HOST'],
    port=os.environ.get('POSTGRES_PORT', '5432'),
    dbname=os.environ['POSTGRES_DB'],
    user=os.environ['POSTGRES_USER'],
    password=os.environ['POSTGRES_PASSWORD'],
    connect_timeout=15,
    options='-c default_transaction_read_only=on -c statement_timeout=60000',
)
connection.set_session(readonly=True)
with connection, connection.cursor() as cursor:
    cursor.execute('SELECT current_database(), inet_server_addr()::text, inet_server_port(), current_schema(), current_setting(\'transaction_read_only\')')
    identity = cursor.fetchone()
    assert identity[0] == expected, 'Unexpected database identity'
    cursor.execute('SELECT version_num FROM public.alembic_version ORDER BY version_num')
    revisions = [row[0] for row in cursor.fetchall()]
    cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='amendment_proposal' ORDER BY ordinal_position")
    columns = [row[0] for row in cursor.fetchall()]
    counts = {}
    for table in ('user', 'file_record', 'regulatory_chunk', 'amendment_proposal'):
        cursor.execute('SELECT to_regclass(%s)', ('public.' + ('"user"' if table == 'user' else table),))
        if cursor.fetchone()[0]:
            cursor.execute('SELECT count(*) FROM public."' + table + '"')
            counts[table] = cursor.fetchone()[0]

migration_map = {}
for path in Path('/app/alembic/versions').glob('*.py'):
    values = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in ('revision', 'down_revision'):
                values[node.target.id] = ast.literal_eval(node.value)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('revision', 'down_revision'):
                    values[target.id] = ast.literal_eval(node.value)
    if 'revision' in values:
        migration_map[values['revision']] = {'parent': values.get('down_revision'), 'file': path.name}
print(json.dumps({'database': identity[0], 'server': identity[1], 'port': identity[2], 'schema': identity[3], 'read_only': identity[4], 'image_version': os.environ.get('ONYX_VERSION'), 'db_revisions': revisions, 'current_revisions_in_image': {revision: revision in migration_map for revision in revisions}, 'amendment_columns': columns, 'counts': counts, 'migrations': migration_map}, sort_keys=True))
