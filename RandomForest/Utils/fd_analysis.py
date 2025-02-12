import pandas as pd
from fdas.fd import discover_fd

# Load CSV
df = pd.read_csv("TestData/realData/iris.csv")

# Convert DataFrame to a list of tuples
data = [tuple(row) for row in df.to_numpy()]

# Get functional dependencies
fds = discover_fd(data, header=df.columns)

# Print results
for determinant, dependents in fds.items():
    print(f"✅ Functional Dependency Found: {determinant} → {', '.join(dependents)}")