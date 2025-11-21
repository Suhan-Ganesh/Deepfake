#!/usr/bin/env python3
"""
Test script to verify Sepolia testnet connection and blockchain functionality
"""

import os
import sys
from dotenv import load_dotenv
from web3 import Web3

def test_blockchain_connection():
    """Test the blockchain connection"""
    print("Testing Sepolia testnet connection...")
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    infura_url = os.getenv("INFURA_URL")
    contract_address = os.getenv("CONTRACT_ADDRESS")
    account_address = os.getenv("ACCOUNT_ADDRESS")
    private_key = os.getenv("PRIVATE_KEY")
    
    print(f"INFURA_URL: {infura_url}")
    print(f"CONTRACT_ADDRESS: {contract_address}")
    print(f"ACCOUNT_ADDRESS: {account_address}")
    print(f"PRIVATE_KEY: {'SET' if private_key else 'NOT SET'}")
    
    # Check if all required values are present
    if not all([infura_url, contract_address, account_address, private_key]):
        print("❌ Missing required configuration values")
        return False
    
    # Connect to blockchain
    web3 = Web3(Web3.HTTPProvider(infura_url))
    
    if not web3.is_connected():
        print("❌ Failed to connect to Sepolia testnet")
        return False
    
    print("✅ Connected to Sepolia testnet successfully")
    
    # Check account balance
    try:
        account_balance = web3.eth.get_balance(account_address)
        print(f"Account balance: {web3.from_wei(account_balance, 'ether')} ETH")
    except Exception as e:
        print(f"❌ Error getting account balance: {e}")
        return False
    
    # Check if account has enough balance for transactions
    if account_balance == 0:
        print("⚠️ Account has zero balance. You'll need some Sepolia ETH to perform transactions.")
        print("Get Sepolia ETH from: https://sepoliafaucet.com/")
        return True  # Connection works, but no funds
    
    return True

def test_contract_interaction():
    """Test contract interaction"""
    print("\nTesting contract interaction...")
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    infura_url = os.getenv("INFURA_URL")
    contract_address = os.getenv("CONTRACT_ADDRESS")
    account_address = os.getenv("ACCOUNT_ADDRESS")
    private_key = os.getenv("PRIVATE_KEY")
    
    # Connect to blockchain
    web3 = Web3(Web3.HTTPProvider(infura_url))
    
    if not web3.is_connected():
        print("❌ Not connected to blockchain")
        return False
    
    # Define contract ABI
    abi = [
        {
            "inputs": [
                {"internalType": "string", "name": "_fileHash", "type": "string"}
            ],
            "name": "registerMedia",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [],
            "name": "getAllRecords",
            "outputs": [
                {
                    "components": [
                        {"internalType": "string", "name": "fileHash", "type": "string"},
                        {"internalType": "address", "name": "uploader", "type": "address"},
                        {"internalType": "uint256", "name": "timestamp", "type": "uint256"}
                    ],
                    "internalType": "struct DeepfakeAuth.MediaRecord[]",
                    "name": "",
                    "type": "tuple[]"
                }
            ],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [],
            "name": "totalRecords",
            "outputs": [
                {"internalType": "uint256", "name": "", "type": "uint256"}
            ],
            "stateMutability": "view",
            "type": "function"
        }
    ]
    
    # Create contract instance
    try:
        contract = web3.eth.contract(address=contract_address, abi=abi)
        print("✅ Contract instance created successfully")
    except Exception as e:
        print(f"❌ Error creating contract instance: {e}")
        return False
    
    # Test reading from contract
    try:
        total_records = contract.functions.totalRecords().call()
        print(f"✅ Total records in contract: {total_records}")
    except Exception as e:
        print(f"❌ Error calling totalRecords: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("Sepolia Testnet Blockchain Test")
    print("=" * 40)
    
    # Test connection
    if not test_blockchain_connection():
        return False
    
    # Test contract interaction
    if not test_contract_interaction():
        return False
    
    print("\n✅ All tests passed! Blockchain is properly configured.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)