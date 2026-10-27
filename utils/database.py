import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()


class DatabaseUtil:

    def __init__(self, db_config):
        self.db_config = db_config

        try: 
            self.connection = psycopg2.connect(**db_config) 

        except Exception as e:
            print(f"Error connecting to the database: {e}")
            self.connection = None

    def schema_details(self,schema_name):

        schema_info_context = ""
        
        connection = self.connection
        cursor = connection.cursor()

        schema_info_context = f"Database Schema: {schema_name}\n"

        try: 

            cursor.execute("SELECT table_name from information_schema.tables where table_schema = %s;", (schema_name,))
            tables_list = cursor.fetchall()

            for table in tables_list:
                table_name = table[0]
                schema_info_context = f"{schema_info_context}\nTable: {table_name}\n"

                column_constraints = self._column_constraints(cursor, schema_name, table_name)
                column_indexes = self._column_indexes(cursor, schema_name, table_name)

                # Adding Columns, Data Types, Constraints & Indexes
                # ordinal_position keeps columns in the same order as the sample data rows
                cursor.execute(
                    """
                    SELECT column_name, data_type, character_maximum_length, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position;
                    """,
                    (schema_name, table_name),
                )
                columns_list = cursor.fetchall()

                for column in columns_list:
                    column_name, data_type, max_length, is_nullable, column_default = column
                    if max_length:
                        data_type = f"{data_type}({max_length})"

                    schema_info_context = f"{schema_info_context}  Column: {column_name}, Data Type: {data_type}, Nullable: {is_nullable}"
                    if column_default:
                        schema_info_context = f"{schema_info_context}, Default: {column_default}"
                    schema_info_context = f"{schema_info_context}\n"

                    if column_name in column_constraints:
                        schema_info_context = f"{schema_info_context}    Constraints: {'; '.join(column_constraints[column_name])}\n"
                    if column_name in column_indexes:
                        schema_info_context = f"{schema_info_context}    Indexes: {'; '.join(column_indexes[column_name])}\n"

                # Indexes on expressions (e.g. lower(email)) don't map to a single column
                for index_def in column_indexes.get(None, []):
                    schema_info_context = f"{schema_info_context}  Expression Index: {index_def}\n"

                # Adding Sample Data
                cursor.execute(f"SELECT * FROM {schema_name}.{table_name} LIMIT 5;")
                sample_data = cursor.fetchall()
                schema_info_context = f"{schema_info_context}  Sample Data:\n"
                for row in sample_data:
                    schema_info_context = f"{schema_info_context}    {row}\n"

        except Exception as e:
            print(f"Error fetching schema details: {e}")
            schema_info_context = f"Error fetching schema details: {e}"

        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
        
        return schema_info_context

    def _column_constraints(self, cursor, schema_name, table_name):
        """Returns {column_name: [constraint descriptions]} for a table."""

        # pg_catalog is used instead of information_schema because the latter
        # hides constraints on tables the current role doesn't own.
        cursor.execute(
            """
            SELECT con.conname,
                   pg_get_constraintdef(con.oid),
                   ARRAY(
                       SELECT att.attname::text
                       FROM unnest(con.conkey) AS k(attnum)
                       JOIN pg_attribute att ON att.attrelid = con.conrelid AND att.attnum = k.attnum
                   )
            FROM pg_constraint con
            JOIN pg_class rel ON rel.oid = con.conrelid
            JOIN pg_namespace nsp ON nsp.oid = rel.relnamespace
            WHERE nsp.nspname = %s AND rel.relname = %s;
            """,
            (schema_name, table_name),
        )

        column_constraints = {}
        for constraint_name, constraint_def, column_names in cursor.fetchall():
            for column_name in column_names:
                column_constraints.setdefault(column_name, []).append(f"{constraint_name}: {constraint_def}")

        return column_constraints

    def _column_indexes(self, cursor, schema_name, table_name):
        """Returns {column_name: [index definitions]} for a table. Expression-only indexes are keyed by None."""

        cursor.execute(
            """
            SELECT idx.relname,
                   pg_get_indexdef(ind.indexrelid),
                   ARRAY(
                       SELECT att.attname::text
                       FROM unnest(ind.indkey) AS k(attnum)
                       JOIN pg_attribute att ON att.attrelid = ind.indrelid AND att.attnum = k.attnum
                   )
            FROM pg_index ind
            JOIN pg_class idx ON idx.oid = ind.indexrelid
            JOIN pg_class rel ON rel.oid = ind.indrelid
            JOIN pg_namespace nsp ON nsp.oid = rel.relnamespace
            WHERE nsp.nspname = %s AND rel.relname = %s;
            """,
            (schema_name, table_name),
        )

        column_indexes = {}
        for index_name, index_def, column_names in cursor.fetchall():
            # Strip the "CREATE [UNIQUE] INDEX name ON schema.table" prefix, keep "USING btree (cols)"
            index_summary = index_def[index_def.index(" USING "):].strip()
            if " UNIQUE " in f" {index_def} ":
                index_summary = f"UNIQUE {index_summary}"
            for column_name in column_names or [None]:
                column_indexes.setdefault(column_name, []).append(f"{index_name} ({index_summary})")

        return column_indexes

    def execute_sql(self, query):
        try:
            connection = self.connection
            cursor = connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            connection.commit()
            return str(result)
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()



obj = DatabaseUtil({
    "host": os.environ['host'],
    "database": os.environ['database'],
    "user": os.environ['user'],
    "password": os.environ['password'],
    "port": int(os.environ['port'])
})

result = obj.schema_details("public")

with open("test_schema_details.txt", "w") as f:
    f.write(result)