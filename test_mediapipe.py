"""
Test script to diagnose MediaPipe installation issues
"""
import sys

print("Python version:", sys.version)
print("\nTesting MediaPipe installation...")

try:
    import mediapipe as mp
    print("✓ MediaPipe imported successfully")
    
    # Check version
    try:
        print(f"✓ MediaPipe version: {mp.__version__}")
    except AttributeError:
        print("⚠ MediaPipe version not available")
    
    # Check for solutions module
    print("\nChecking MediaPipe attributes...")
    attrs = [attr for attr in dir(mp) if not attr.startswith('_')]
    print(f"Available attributes: {attrs}")
    
    if hasattr(mp, 'solutions'):
        print("✓ 'solutions' module found")
        try:
            mp_hands = mp.solutions.hands
            print("✓ 'solutions.hands' accessible")
            
            mp_drawing = mp.solutions.drawing_utils
            print("✓ 'solutions.drawing_utils' accessible")
            
            print("\n✓ MediaPipe is working correctly!")
        except AttributeError as e:
            print(f"✗ Error accessing solutions: {e}")
    else:
        print("✗ 'solutions' module NOT found")
        print("\nThis is likely a MediaPipe installation issue.")
        print("Try reinstalling:")
        print("  pip uninstall mediapipe")
        print("  pip install mediapipe>=0.10.8")
        
except ImportError as e:
    print(f"✗ Failed to import MediaPipe: {e}")
    print("\nPlease install MediaPipe:")
    print("  pip install mediapipe>=0.10.8")
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
