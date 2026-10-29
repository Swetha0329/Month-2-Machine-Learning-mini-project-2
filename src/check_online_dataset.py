import requests

url = "https://datasets-server.huggingface.co/first-rows"

params = {
    "dataset": "zefang-liu/phishing-email-dataset",
    "config": "default",
    "split": "train"
}

try:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    result = response.json()

    print("DATASET ACCESS SUCCESSFUL")
    print("\nCOLUMNS:")

    for column in result.get("features", []):
        print("-", column["name"])

    print("\nSAMPLE MESSAGES:")

    for row in result.get("rows", [])[:3]:
        print(row["row"])

except requests.RequestException as error:
    print("Could not retrieve sample rows:", error)