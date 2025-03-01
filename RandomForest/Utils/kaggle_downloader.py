import subprocess

def list_kaggle_datasets(search_term):
    try:
        # Run the Kaggle API command
        result = subprocess.run(
            ["kaggle", "datasets", "list", "-s", search_term],
            capture_output=True,
            text=True
        )

        # Print the output
        if result.returncode == 0:
            print(f"\nAvailable datasets for '{search_term}':\n")
            print(result.stdout)
        else:
            print("Error: Kaggle API failed.")
            print(result.stderr)

    except FileNotFoundError:
        print("Kaggle CLI is not installed. Install it using: pip install kaggle")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Example usage: Search for "pokemon"
search_query = input("Enter a dataset name to search: ")
list_kaggle_datasets(search_query)