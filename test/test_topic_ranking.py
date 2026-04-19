import requests
import json
import time

def run_test_case(style, topic, texts):
    url = "http://localhost:1337/rank"
    
    payload = {
        "style_name": style,
        "topic": topic,
        "texts": texts
    }
    
    print(f"{'='*50}")
    print(f"Testing Style: {style}, Topic: {topic}")
    print(f"{'='*50}")
    print("Input Texts:")
    for i, t in enumerate(texts):
        print(f"  [{i}]: {t}")
        
    try:
        start_time = time.time()
        response = requests.post(url, json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print("\n--- Ranking Result ---")
            print(f"Time taken: {end_time - start_time:.2f}s")
            print(f"Raw Output: {result.get('raw_output')}")
            print("\nRanked Order (Best to Worst):")
            for rank, idx in enumerate(result['ranked_indices']):
                print(f"  #{rank+1} (Index {idx}): {result['ranked_texts'][rank]}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("Could not connect to server. Is it running on port 1337?")


def main():
    test_cases = [
        {
            "style": "Aggressive",
            "topic": "Astrophysics",
            "texts": [
                "The damn black hole is sucking everything in like a cosmic vacuum cleaner from hell!", # Aggressive + Astrophysics (Best)
                "The black hole absorbs matter from its surroundings.", # Neutral + Astrophysics (Middle)
                "Hey idiot, get out of my way!", # Aggressive + General (Middle)
                "The weather is nice today." # Neutral + General (Worst)
            ]
        }
    ]
    
    for case in test_cases:
        run_test_case(case["style"], case["topic"], case["texts"])

if __name__ == "__main__":
    main()
