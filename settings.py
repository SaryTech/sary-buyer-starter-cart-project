from decouple import config

SNOWFLAKE_USER = config('SNOWFLAKE_USER')
SNOWFLAKE_PASSWORD = config('SNOWFLAKE_PASSWORD')
SNOWFLAKE_ACCOUNT = config('SNOWFLAKE_ACCOUNT')
SNOWFLAKE_ROLE = config('SNOWFLAKE_ROLE')
SNOWFLAKE_WAREHOUSE = config('SNOWFLAKE_WAREHOUSE')
SNOWFLAKE_DATABASE = config('SNOWFLAKE_DATABASE')
SNOWFLAKE_SCHEMA = config('SNOWFLAKE_SCHEMA')