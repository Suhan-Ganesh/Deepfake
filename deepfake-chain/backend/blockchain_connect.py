from web3 import Web3
import hashlib
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Debug: Print loaded environment variables
# import os
# print("DEBUG: Loaded INFURA_URL:", os.getenv("INFURA_URL", "NOT SET"))
# print("DEBUG: Loaded CONTRACT_ADDRESS:", os.getenv("CONTRACT_ADDRESS", "NOT SET"))
# print("DEBUG: Loaded ACCOUNT_ADDRESS:", os.getenv("ACCOUNT_ADDRESS", "NOT SET"))
# print("DEBUG: Loaded PRIVATE_KEY:", "SET" if os.getenv("PRIVATE_KEY") else "NOT SET")

# ------------------------------------
# 🔗 INFURA / METAMASK CONFIGURATION
# ------------------------------------
INFURA_URL = os.getenv("INFURA_URL", "")
CONTRACT_ADDRESS_STR = os.getenv("CONTRACT_ADDRESS", "")
ACCOUNT_ADDRESS_STR = os.getenv("ACCOUNT_ADDRESS", "")
PRIVATE_KEY = os.getenv("PRIVATE_KEY", "")

# Debug print to see what values are loaded
# print(f"DEBUG: INFURA_URL={INFURA_URL}")
# print(f"DEBUG: CONTRACT_ADDRESS_STR={CONTRACT_ADDRESS_STR}")
# print(f"DEBUG: ACCOUNT_ADDRESS_STR={ACCOUNT_ADDRESS_STR}")
# print(f"DEBUG: PRIVATE_KEY={'SET' if PRIVATE_KEY else 'NOT SET'}")

# Check if blockchain is configured
BLOCKCHAIN_ENABLED = bool(INFURA_URL and CONTRACT_ADDRESS_STR and ACCOUNT_ADDRESS_STR and PRIVATE_KEY)

# Additional check to ensure we're not using placeholder values
if CONTRACT_ADDRESS_STR.startswith("0xYOUR_") or ACCOUNT_ADDRESS_STR.startswith("0xYOUR_") or INFURA_URL.endswith("YOUR_PROJECT_ID"):
    BLOCKCHAIN_ENABLED = False

if BLOCKCHAIN_ENABLED:
    try:
        CONTRACT_ADDRESS = Web3.to_checksum_address(CONTRACT_ADDRESS_STR)
        ACCOUNT_ADDRESS = Web3.to_checksum_address(ACCOUNT_ADDRESS_STR)
    except Exception as e:
        print(f"❌ Error processing blockchain addresses: {e}")
        BLOCKCHAIN_ENABLED = False
    
    # Connect to blockchain
    web3 = Web3(Web3.HTTPProvider(INFURA_URL))
    if web3.is_connected():
        print("✅ Connected to Ethereum Sepolia Testnet")
    else:
        print("❌ Connection failed")
        BLOCKCHAIN_ENABLED = False
else:
    print("⚠️ Blockchain not configured. Running in local mode.")
    print("   To enable blockchain, configure INFURA_URL, CONTRACT_ADDRESS, ACCOUNT_ADDRESS, and PRIVATE_KEY")
    web3 = None
    CONTRACT_ADDRESS = None
    ACCOUNT_ADDRESS = None

# ------------------------------------
# 🔐 CONTRACT ABI (copy from Remix)
# ------------------------------------
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

# Get contract instance if blockchain is enabled
if BLOCKCHAIN_ENABLED:
    contract = web3.eth.contract(address=CONTRACT_ADDRESS, abi=abi)
    web3.eth.default_account = ACCOUNT_ADDRESS
else:
    contract = None

# ------------------------------------
# 🧠 Upload function (called by Flask)
# ------------------------------------
def upload_to_blockchain(file_data, blockchain_data=None):
    try:
        # Hash file using SHA-256
        file_hash = hashlib.sha256(file_data).hexdigest()
        
        # Prepare data string for blockchain
        if blockchain_data:
            data_str = f"{blockchain_data.get('filename', 'unknown')}|{file_hash}|deepfake:{blockchain_data.get('is_deepfake', False)}|confidence:{blockchain_data.get('confidence', 0.0)}|type:{blockchain_data.get('file_type', 'unknown')}"
        else:
            data_str = file_hash
        
        # If blockchain is not configured, return mock transaction hash
        if not BLOCKCHAIN_ENABLED:
            print(f"📌 Local mode - File hash: {file_hash}")
            if blockchain_data:
                print(f"📊 Deepfake: {blockchain_data.get('is_deepfake')} (Confidence: {blockchain_data.get('confidence')})")
            mock_tx_hash = f"0x{hashlib.sha256(data_str.encode()).hexdigest()}"
            print(f"🔗 Mock transaction hash: {mock_tx_hash}")
            return mock_tx_hash

        # Create transaction
        nonce = web3.eth.get_transaction_count(ACCOUNT_ADDRESS)
        txn = contract.functions.registerMedia(data_str).build_transaction({
            'from': ACCOUNT_ADDRESS,
            'nonce': nonce,
            'gas': 300000,
            'gasPrice': web3.to_wei('20', 'gwei')
        })

        # Sign and send transaction
        signed_txn = web3.eth.account.sign_transaction(txn, private_key=PRIVATE_KEY)
        tx_hash = web3.eth.send_raw_transaction(signed_txn.rawTransaction)


        print(f"✅ Uploaded hash: {file_hash}")
        if blockchain_data:
            print(f"📊 Deepfake: {blockchain_data.get('is_deepfake')} (Confidence: {blockchain_data.get('confidence')})")
        print(f"🔗 Transaction hash: {web3.to_hex(tx_hash)}")

        return web3.to_hex(tx_hash)

    except Exception as e:
        print("❌ Upload error:", str(e))
        raise


# ------------------------------------
# 🔍 View blockchain data (called by Flask)
# ------------------------------------
def view_blockchain():
    try:
        # If blockchain is not configured, return empty list
        if not BLOCKCHAIN_ENABLED:
            print("📌 Local mode - No blockchain data available")
            return []
        
        # Fetch all records
        records = contract.functions.getAllRecords().call()

        # Decode properly
        result = []
        for r in records:
            file_hash = r[0]
            uploader = r[1]
            timestamp = r[2]
            result.append({
                "fileHash": file_hash,
                "uploader": web3.to_checksum_address(uploader),
                "timestamp": int(timestamp)
            })

        print(f"📦 Blockchain contains {len(result)} records.")
        return result

    except ValueError as err:
        # Handle revert errors gracefully
        if "execution reverted" in str(err):
            print("⚠️ Contract returned 'execution reverted' — possibly empty or inaccessible data.")
            return []
        else:
            print("❌ View blockchain error:", str(err))
            raise

    except Exception as e:
        print("❌ View blockchain error:", str(e))
        raise

# ------------------------------------
# 🔢 Get total records count
# ------------------------------------
def get_total_records():
    try:
        # If blockchain is not configured, return 0
        if not BLOCKCHAIN_ENABLED:
            return 0
        
        # Fetch total records count
        count = contract.functions.totalRecords().call()
        return int(count)
    except Exception as e:
        print("❌ Get total records error:", str(e))
        return 0

