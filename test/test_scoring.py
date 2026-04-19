"""
Test client for the /score endpoint.
Tests the new pointwise scoring system with sample texts.
"""
import requests
import json

SERVER_URL = "http://localhost:1337/score"

def test_score(style: str, topic: str, text: str, expected_approx: str = ""):
    """Test scoring a single text."""
    print(f"\n{'='*60}")
    print(f"Topic: {topic}")
    print(f"Style: {style}")
    print(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
    if expected_approx:
        print(f"Expected: {expected_approx}")
    print("-" * 40)
    
    try:
        response = requests.post(SERVER_URL, json={
            "style_name": style,
            "topic": topic,
            "text": text
        }, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        print(f"Final Score: {result['final_score']:.3f}")
        print(f"Criteria:")
        c = result['criteria']
        print(f"  topic_relevant: {c['topic_relevant']}")
        print(f"  style_match: {c['style_match']:.2f}")
        print(f"  topic_depth: {c['topic_depth']:.2f}")
        print(f"  topic_style_coherence: {c['topic_style_coherence']:.2f}")
        print(f"  creativity_impact: {c['creativity_impact']:.2f}")
        print(f"Raw output: {result.get('raw_output', 'N/A')}")
        
        return result
        
    except requests.exceptions.Timeout:
        print(f"Error: Request timed out (model may still be loading)")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def wait_for_server(max_retries=30, delay=2):
    """Wait for server to be ready."""
    import time
    print("Waiting for server to be ready...")
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:1337/docs", timeout=5)
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except:
            pass
        print(f"  Attempt {i+1}/{max_retries} - server not ready, waiting {delay}s...")
        time.sleep(delay)
    print("Server did not become ready in time.")
    return False

def main():
    print("Testing /score endpoint...")
    print("Make sure server is running: python rank_server.py")
    
    if not wait_for_server():
        return
    
    # Test 1: Perfect match - aggressive astrophysics
    test_score(
        style="aggressive",
        topic="astrophysics", 
        text="Damn it, these idiotic skeptics still don't understand that black holes aren't some abstract nonsense! The singularity will crush matter into a point, and your pathetic physics won't save you! The event horizon is a one-way ticket, you morons!",
        expected_approx="high (0.6-0.8) - matches both"
    )
    
    # Test 2: Topic match, no style (neutral academic)
    test_score(
        style="aggressive",
        topic="astrophysics",
        text="Black holes form from the collapse of massive stars. Their gravitational field is so strong that even light cannot escape from within the event horizon. The Schwarzschild radius defines this boundary.",
        expected_approx="medium (0.3-0.5) - topic but neutral style"
    )
    
    # Test 3: Style match, WRONG topic - should be ZERO
    test_score(
        style="aggressive",
        topic="astrophysics",
        text="What the hell are you doing in the kitchen?! This dish is complete garbage! Have you never seen how to properly fry meat?! Throw this trash out and start over, you incompetent fool!",
        expected_approx="ZERO (0.0) - wrong topic, must be 0"
    )
    
    # Test 4: No match at all - should be ZERO
    test_score(
        style="aggressive",
        topic="astrophysics",
        text="Today was a pleasant day. I took a walk in the park, fed the ducks, and had a cup of tea. The weather was wonderful.",
        expected_approx="ZERO (0.0) - completely off topic"
    )
    
    # Test 5: Weak topic connection with strong style
    test_score(
        style="aggressive",
        topic="astrophysics",
        text="Stars, damn it! Everyone looks at these stupid lights in the sky and thinks they understand something! You don't know a damn thing about the cosmos and how neutron stars form!",
        expected_approx="medium (0.4-0.6) - has topic + style but shallow"
    )
    
    # Test 6: Deep topic, opposite style (gentle/poetic)
    test_score(
        style="aggressive",
        topic="astrophysics",
        text="The gentle dance of galaxies across the cosmic tapestry reminds us of our humble place in the universe. Quasars, those distant beacons, whisper ancient secrets of supermassive black holes.",
        expected_approx="low-medium (0.2-0.4) - topic yes, style mismatch"
    )

if __name__ == "__main__":
    main()
