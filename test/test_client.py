import requests
import json
import time

def run_test_case(style, texts):
    url = "http://localhost:1337/rank"
    
    payload = {
        "style_name": style,
        "texts": texts
    }
    
    print(f"{'='*50}")
    print(f"Testing Style: {style}")
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
            "style": "Formal Scientific",
            "texts": [
                "Hey dude, check out this cool rock I found!",
                "The specimen collected from the site exhibits crystalline structures consistent with quartz.",
                "I think this stone is pretty neat, it's shiny."
            ]
        },
        {
            "style": "Shakespearean / Poetic",
            "texts": [
                "I really miss you and I hope you are doing well.",
                "To be, or not to be, that is the question.",
                "Hark! What light through yonder window breaks? It is the east, and Juliet is the sun.",
                "Yo, what's up? Long time no see."
            ]
        },
        {
            "style": "Clickbait YouTube Title",
            "texts": [
                "A tutorial on how to bake a cake.",
                "YOU WON'T BELIEVE WHAT HAPPENED WHEN I BAKED THIS CAKE! (GONE WRONG)",
                "Baking a chocolate cake: step by step guide.",
                "I baked a cake and it was okay."
            ]
        },
        {
            "style": "Passive Aggressive Corporate Email",
            "texts": [
                "You are an idiot for missing the deadline.",
                "Per my last email, please ensure the deadline is met next time to avoid delays.",
                "Please try to finish on time next time.",
                "I'm just checking in on this, as I haven't heard back yet."
            ]
        },
        {
            "style": "Pirate Speak",
            "texts": [
                "Hello friend, would you like to go sailing?",
                "Ahoy matey! Hoist the colors and set sail for the seven seas!",
                "Let's get on the boat and go.",
                "Yarrr! Shiver me timbers, there be treasure ahead!"
            ]
        }
    ]

    for case in test_cases:
        run_test_case(case["style"], case["texts"])
        time.sleep(1) # Small pause between requests

if __name__ == "__main__":
    main()
