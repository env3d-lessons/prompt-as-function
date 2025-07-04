import sys
from prompt_function import PromptFunction, timer

# Parse model index from command-line argument
model_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

prompt = "Categorize the merchant type. Output exactly one full English word, lowercase, fully spelled out, no abbreviations, no partial words, and no punctuation.  Categories: grocery, retail, restaurant, pharmacy, entertainment, transportation, clothing, lodging, electronics, food, delivery, telecom, services:"

extract_category = PromptFunction(prompt, model_index)

merchants = [
    "Amazon",
    "Starbucks",
    "Walmart",
    "Target",
    "Apple Store",
    "Costco",
    "Uber",
    "McDonald's",
    "Netflix",
    "Best Buy",
    "Shell",
    "CVS Pharmacy",
    "Home Depot",
    "Walgreens",
    "Nike",
    "Subway",
    "Delta Airlines",
    "Spotify",
    "Lowe's",
    "Chipotle",
    "Airbnb",
    "FedEx",
    "Whole Foods Market",
    "H&M",
    "Google Play",
    "AT&T",
    "IKEA",
    "Domino's Pizza",
    "Burger King",
    "eBay"
]

print("| Merchant | Category | Elapse Time (ms) |")
print("| -------- | -------- | ---------------- |")
for merchant in merchants:
    with timer() as get_time:
        category = extract_category(merchant)    
    print(f"| {merchant} | {category} | {get_time():.2f} |")
