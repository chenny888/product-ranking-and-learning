#!/usr/bin/env python3
"""
Validation script to test that all dependencies are working correctly
and core functions can be imported and run.
"""

import sys
import traceback

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
        
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
        
        import pandas as pd
        print("✓ pandas imported successfully")
        
        import scipy
        from scipy.stats import bernoulli
        print("✓ scipy imported successfully")
        
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import normalize
        print("✓ scikit-learn imported successfully")
        
        from utils import AssortOpt, ApproxOpt, TrueReward, UpperBound
        print("✓ utils module imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        traceback.print_exc()
        return False

def test_core_functions():
    """Test that core functions work with simple examples."""
    print("\nTesting core functions...")
    
    try:
        import numpy as np
        from utils import AssortOpt, ApproxOpt, TrueReward
        
        # Simple test case
        n = 5  # number of products
        r = np.array([10, 8, 6, 4, 2])  # revenues
        p = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # choice probabilities
        M = 3  # attention span
        G = [0.95 ** i for i in range(M)]  # attention decay
        
        # Test AssortOpt
        sigma = AssortOpt(r, p, M)
        print(f"✓ AssortOpt completed: found {len(sigma)} solutions")
        
        # Test ApproxOpt
        result, lower_bound, H, sigma_full = ApproxOpt(r, p, G)
        print(f"✓ ApproxOpt completed: ranking {result}, lower bound {lower_bound:.3f}")
        
        # Test TrueReward
        reward = TrueReward(result, r[result], p[result], G)
        print(f"✓ TrueReward completed: reward {reward:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Function test error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("Revenue Management Experiments - Environment Validation")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Import tests failed. Please check your environment setup.")
        sys.exit(1)
    
    # Test core functions
    functions_ok = test_core_functions()
    
    if not functions_ok:
        print("\n❌ Function tests failed. Please check the utils.py file.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 All validation tests passed! Environment is ready.")
    print("\nNext steps:")
    print("1. Open Jupyter notebook: uv run jupyter notebook")
    print("2. Run Offline_Experiments_Section_6_1.ipynb")
    print("3. Run online experiments: uv run python Online_experiment_Section_6_2.py")

if __name__ == "__main__":
    main()
