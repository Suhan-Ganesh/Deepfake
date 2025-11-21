#!/usr/bin/env python3
"""
Debug script to check environment variable loading
"""

import os
from dotenv import load_dotenv

def debug_env():
    """Debug environment variables"""
    print("Current working directory:", os.getcwd())
    print("Files in current directory:")
    for file in os.listdir('.'):
        print(f"  {file}")
    
    print("\nChecking .env file:")
    if os.path.exists('.env'):
        print("  .env file exists")
        with open('.env', 'r') as f:
            content = f.read()
            print("  .env content:")
            for line in content.split('\n'):
                if line.strip() and not line.strip().startswith('#'):
                    print(f"    {line}")
    else:
        print("  .env file does not exist")
    
    print("\nLoading environment variables...")
    load_dotenv()
    
    print("\nLoaded environment variables:")
    infura_url = os.getenv("INFURA_URL")
    contract_address = os.getenv("CONTRACT_ADDRESS")
    account_address = os.getenv("ACCOUNT_ADDRESS")
    private_key = os.getenv("PRIVATE_KEY")
    
    print(f"  INFURA_URL: {infura_url}")
    print(f"  CONTRACT_ADDRESS: {contract_address}")
    print(f"  ACCOUNT_ADDRESS: {account_address}")
    print(f"  PRIVATE_KEY: {'SET' if private_key else 'NOT SET'}")
    
    # Check if values are placeholder values
    if infura_url and infura_url.endswith("YOUR_PROJECT_ID"):
        print("  ⚠️  INFURA_URL contains placeholder value")
    if contract_address and contract_address.startswith("0xYOUR_"):
        print("  ⚠️  CONTRACT_ADDRESS contains placeholder value")
    if account_address and account_address.startswith("0xYOUR_"):
        print("  ⚠️  ACCOUNT_ADDRESS contains placeholder value")

if __name__ == "__main__":
    debug_env()