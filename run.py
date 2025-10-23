#!/usr/bin/env python3
"""
Simplified runner for Student Performance Predictor
"""

import os
import sys
import subprocess

def main():
    """Run the Streamlit application"""
    try:
        # Check if requirements are installed
        try:
            import streamlit
            import pandas
            import sklearn
        except ImportError:
            print("❌ Required packages not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        # Run the Streamlit app
        app_path = os.path.join("app", "student_ml_dashboard.py")
        if os.path.exists(app_path):
            print("🚀 Starting Student Performance Predictor...")
            print("📊 Open http://localhost:8501 in your browser")
            subprocess.run(["streamlit", "run", app_path])
        else:
            print(f"❌ Application not found at {app_path}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()