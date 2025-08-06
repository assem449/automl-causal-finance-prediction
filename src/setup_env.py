#!/usr/bin/env python3
"""
Setup script to help create the .env file with your FRED API key.
"""

import os
import sys

def create_env_file():
    """Create a .env file with user input."""
    
    print("🔧 FRED API Key Setup")
    print("="*40)
    
    # Check if .env already exists
    if os.path.exists('.env'):
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if response != 'y':
            print("Setup cancelled.")
            return
    
    print("\nTo get your free FRED API key:")
    print("1. Go to: https://fred.stlouisfed.org/docs/api/api_key.html")
    print("2. Sign up for a free account")
    print("3. Generate an API key")
    print()
    
    # Get API key from user
    api_key = input("Enter your FRED API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Setup cancelled.")
        return
    
    # Get start date (optional)
    start_date = input("Enter start date for data (YYYY-MM-DD) [default: 1990-01-01]: ").strip()
    if not start_date:
        start_date = "1990-01-01"
    
    # Create .env file content
    env_content = f"""# FRED API Configuration
# Get your free API key from: https://fred.stlouisfed.org/docs/api/api_key.html

FRED_API_KEY={api_key}

# Optional: Customize data fetch settings
START_DATE={start_date}
"""
    
    # Write to .env file
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("\n✅ .env file created successfully!")
        print("📁 File location: .env")
        print("\nYou can now run: python fetch_fred_data.py")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return

def test_api_key():
    """Test if the API key works."""
    print("\n🧪 Testing API key...")
    
    try:
        from dotenv import load_dotenv
        from fredapi import Fred
        
        # Load environment variables
        load_dotenv()
        
        # Get API key
        api_key = os.getenv('FRED_API_KEY')
        
        if not api_key:
            print("❌ No API key found in .env file")
            return False
        
        # Test connection
        fred = Fred(api_key=api_key)
        
        # Try to fetch a simple series
        test_data = fred.get_series('FEDFUNDS', limit=1)
        
        if test_data is not None and len(test_data) > 0:
            print("✅ API key is working correctly!")
            return True
        else:
            print("❌ API key test failed - no data returned")
            return False
            
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        return False

def main():
    """Main setup function."""
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Test existing API key
        test_api_key()
        return
    
    # Create .env file
    create_env_file()
    
    # Ask if user wants to test the API key
    if os.path.exists('.env'):
        response = input("\nWould you like to test your API key? (Y/n): ").strip().lower()
        if response != 'n':
            test_api_key()

if __name__ == "__main__":
    main() 