"""setup.py for editable installs.

The migration_watchdog package lives in migration_watchdog/ at the repo root.
"""
from setuptools import setup, find_packages

setup(
    name="migration-watchdog",
    version="1.3.0",
    packages=find_packages(include=["migration_watchdog*"]),
    package_data={
        "migration_watchdog": [
            "alias_table.json",
            "cli_command_index.json",
        ]
    },
    python_requires=">=3.10",
    install_requires=[
        "boto3>=1.34",
        "botocore>=1.34",
        "httpx>=0.27",
        "fastapi>=0.111",
        "mangum>=0.17",
        "python-dateutil>=2.9",
        "PyJWT>=2.8",
        "strands-agents>=0.1",
        "strands-agents-tools>=0.1",
    ],
)
