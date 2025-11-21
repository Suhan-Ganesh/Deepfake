#!/usr/bin/env python3
"""
Debug script to check environment variable loading with override detection
"""

import os
from dotenv import load_dotenv

def debug_env():
    """Debug environment variables"""
    print("Checking system environment variables BEFORE loading .env:")
    system_infura = os.environ.get("INFURA_URL")
    system_contract = os.environ.get("CONTRACT_ADDRESS")
    system_account = os.environ.get("ACCOUNT_ADDRESS")
    system_private = os.environ.get("PRIVATE_KEY")
    
    print(f"  INFURA_URL: {system_infura}")
    print(f"  CONTRACT_ADDRESS: {system_contract}")
    print(f"  ACCOUNT_ADDRESS: {system_account}")
    print(f"  PRIVATE_KEY: {'SET' if system_private else 'NOT SET'}")
    
    print("\nLoading .env file...")
    load_dotenv()
    
    print("\nChecking system environment variables AFTER loading .env:")
    system_infura = os.environ.get("INFURA_URL")
    system_contract = os.environ.get("CONTRACT_ADDRESS")
    system_account = os.environ.get("ACCOUNT_ADDRESS")
    system_private = os.environ.get("PRIVATE_KEY")
    
    print(f"  INFURA_URL: {system_infura}")
    print(f"  CONTRACT_ADDRESS: {system_contract}")
    print(f"  ACCOUNT_ADDRESS: {system_account}")
    print(f"  PRIVATE_KEY: {'SET' if system_private else 'NOT SET'}")
    
    print("\nChecking os.getenv values:")
    infura_url = os.getenv("INFURA_URL")
    contract_address = os.getenv("CONTRACT_ADDRESS")
    account_address = os.getenv("ACCOUNT_ADDRESS")
    private_key = os.getenv("PRIVATE_KEY")
    
    print(f"  INFURA_URL: {infura_url}")
    print(f"  CONTRACT_ADDRESS: {contract_address}")
    print(f"  ACCOUNT_ADDRESS: {account_address}")
    print(f"  PRIVATE_KEY: {'SET' if private_key else 'NOT SET'}")

if __name__ == "__main__":
    debug_env()