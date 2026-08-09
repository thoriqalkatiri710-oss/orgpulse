import pandas as pd
from sqlalchemy import create_engine
import yaml

# Load config
with open("configs/db_config.yaml") as f:
    cfg = yaml.safe_load(f)["database"]

db = cfg
engine = create_engine(
    f"postgresql://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['name']}"
)


def load_all():
    # 1. Employees
    print("Loading employees...")
    df = pd.read_csv("data/simulated/employees.csv")
    df.to_sql("employees", engine, if_exists="append", index=False)
    print(f"  ✅ {len(df)} rows")

    # 2. Interactions
    print("Loading interactions...")
    df = pd.read_csv("data/simulated/communications.csv")
    df.to_sql("interactions", engine, if_exists="append", index=False)
    print(f"  ✅ {len(df)} rows")

    # 3. Performance reviews
    print("Loading performance reviews...")
    df = pd.read_csv("data/simulated/performance_reviews.csv")
    df.to_sql("performance_reviews", engine, if_exists="append", index=False)
    print(f"  ✅ {len(df)} rows")

    # 4. Training
    print("Loading training data...")
    df = pd.read_csv("data/simulated/training_data.csv")
    df.to_sql("training_completion", engine, if_exists="append", index=False)
    print(f"  ✅ {len(df)} rows")

    # 5. Attendance (batch karena besar)
    print("Loading attendance (batch)...")
    chunks = pd.read_csv("data/simulated/attendance.csv", chunksize=50_000)
    total = 0
    for chunk in chunks:
        chunk.to_sql("attendance", engine, if_exists="append", index=False)
        total += len(chunk)
        print(f"  Loaded {total:,} rows...")
    print(f"  ✅ {total:,} rows total")

    print("\n✅ Semua data berhasil diload ke PostgreSQL!")


if __name__ == "__main__":
    load_all()