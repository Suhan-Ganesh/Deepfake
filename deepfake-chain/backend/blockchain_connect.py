from web3 import Web3
import hashlib
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# ------------------------------------
# 🔗 INFURA / METAMASK CONFIGURATION
# ------------------------------------
INFURA_URL = os.getenv("INFURA_URL", "")
CONTRACT_ADDRESS_STR = os.getenv("CONTRACT_ADDRESS", "")
ACCOUNT_ADDRESS_STR = os.getenv("ACCOUNT_ADDRESS", "")
PRIVATE_KEY = os.getenv("PRIVATE_KEY", "")

# Check if blockchain is configured properly
BLOCKCHAIN_ENABLED = bool(INFURA_URL and CONTRACT_ADDRESS_STR and ACCOUNT_ADDRESS_STR and PRIVATE_KEY)

# Additional check to avoid placeholder values
if CONTRACT_ADDRESS_STR.startswith("0xYOUR_") or ACCOUNT_ADDRESS_STR.startswith("0xYOUR_") or INFURA_URL.endswith("YOUR_PROJECT_ID"):
    BLOCKCHAIN_ENABLED = False

if BLOCKCHAIN_ENABLED:
    try:
        CONTRACT_ADDRESS = Web3.to_checksum_address(CONTRACT_ADDRESS_STR)
        ACCOUNT_ADDRESS = Web3.to_checksum_address(ACCOUNT_ADDRESS_STR)
    except Exception as e:
        print(f"❌ Error processing blockchain addresses: {e}")
        BLOCKCHAIN_ENABLED = False

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
# 🔐 CONTRACT ABI (loaded from JSON file)
# ------------------------------------
def load_contract_abi():
    """Load contract ABI from JSON file"""
    try:
        with open('contract_abi.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading contract ABI: {e}")
        return []

abi = load_contract_abi()

# Get contract instance if blockchain is enabled
if BLOCKCHAIN_ENABLED:
    contract = web3.eth.contract(address=CONTRACT_ADDRESS, abi=abi)
    web3.eth.default_account = ACCOUNT_ADDRESS
else:
    contract = None

# ------------------------------------
# 🧠 Upload function (called by Flask or other)
# ------------------------------------
def check_duplicate_by_hash(file_hash):
    try:
        if not BLOCKCHAIN_ENABLED:
            return False

        try:
            exists = contract.functions.hashExists(file_hash).call()
            return exists
        except Exception as e:
            if "execution reverted" in str(e):
                return False

        try:
            all_records = contract.functions.getAllRecords().call()
            for record in all_records:
                if record[0] == file_hash:
                    return True
            return False
        except Exception as e:
            if "execution reverted" in str(e):
                return False
            return False

    except Exception:
        return False

def upload_to_blockchain(file_data, blockchain_data=None):
    try:
        file_hash = hashlib.sha256(file_data).hexdigest()

        is_deepfake = False
        confidence_score = 0
        if blockchain_data:
            is_deepfake = blockchain_data.get('is_deepfake', False)
            confidence_score = int(blockchain_data.get('confidence', 0.0) * 10000)

        if not BLOCKCHAIN_ENABLED:
            mock_tx_hash = f"0x{hashlib.sha256(file_hash.encode()).hexdigest()}"
            return mock_tx_hash

        if check_duplicate_by_hash(file_hash):
            return {
                "status": "duplicate",
                "message": f"File with hash {file_hash[:10]}... already exists in blockchain",
                "file_hash": file_hash
            }

        nonce = web3.eth.get_transaction_count(ACCOUNT_ADDRESS)

        try:
            txn = contract.functions.registerMedia(
                file_hash,
                is_deepfake,
                confidence_score
            ).build_transaction({
                'from': ACCOUNT_ADDRESS,
                'nonce': nonce,
                'gas': 3000000,
                'gasPrice': web3.to_wei('20', 'gwei')
            })
        except Exception as e:
            print(f"❌ Error building transaction: {e}")
            try:
                print("🔄 Retrying with higher gas limit...")
                txn = contract.functions.registerMedia(
                    file_hash,
                    is_deepfake,
                    confidence_score
                ).build_transaction({
                    'from': ACCOUNT_ADDRESS,
                    'nonce': nonce,
                    'gas': 3000000,
                    'gasPrice': web3.to_wei('20', 'gwei')
                })
                print(f"📦 Transaction built with higher gas")
            except Exception as retry_error:
                print(f"❌ Retry also failed: {retry_error}")
                mock_tx_hash = f"0x{hashlib.sha256(b'mock_error').hexdigest()}"
                return mock_tx_hash

        signed_txn = web3.eth.account.sign_transaction(txn, private_key=PRIVATE_KEY)
        tx_hash = web3.eth.send_raw_transaction(signed_txn.rawTransaction)

        tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

        if tx_receipt.status == 0:
            mock_tx_hash = f"0x{hashlib.sha256(b'mock_error').hexdigest()}"
            return mock_tx_hash

        return web3.to_hex(tx_hash)

    except Exception as e:
        mock_tx_hash = f"0x{hashlib.sha256(b'mock_error').hexdigest()}"
        return mock_tx_hash

# ------------------------------------
# 🔍 View blockchain data (called by Flask)
# ------------------------------------
def view_blockchain():
    try:
        if not BLOCKCHAIN_ENABLED:
            return []

        total_records = contract.functions.totalRecords().call()
        if total_records == 0:
            return []

        records = []
        try:
            records = contract.functions.getAllRecords().call()
        except Exception:
            try:
                records = contract.functions.getRecordsPaginated(0, min(50, total_records)).call()
            except Exception:
                records = []
                for i in range(total_records):
                    try:
                        record = contract.functions.getRecord(i).call()
                        records.append(record)
                    except Exception:
                        break

        result = []
        for r in records:
            if len(r) < 5:
                continue
            file_hash = r[0] or ""
            is_deepfake = bool(r[1]) if len(r) > 1 else False
            confidence_score = int(r[2]) if len(r) > 2 and r[2] else 0
            uploader = r[3] if len(r) > 3 and r[3] else ACCOUNT_ADDRESS
            timestamp = int(r[4]) if len(r) > 4 and r[4] else 0

            result.append({
                "fileHash": file_hash,
                "isDeepfake": is_deepfake,
                "confidenceScore": confidence_score / 10000.0,
                "uploader": web3.to_checksum_address(uploader),
                "timestamp": timestamp
            })

        return result

    except Exception as e:
        return []

# ------------------------------------
# 🔢 Get total records count
# ------------------------------------
def get_total_records():
    try:
        if not BLOCKCHAIN_ENABLED:
            return 0
        count = contract.functions.totalRecords().call()
        return int(count)
    except Exception as e:
        return 0

# ------------------------------------
# 🚫 Remove circular import and fix fallback
# ------------------------------------
def upload_with_local_fallback(file_data, blockchain_data=None):
    # Save to local storage first
    if blockchain_data:
        try:
            from local_storage import save_local_record
            local_id = save_local_record(blockchain_data)  # Ensure this function exists and imported
        except Exception:
            pass
    
    try:
        result = upload_to_blockchain(file_data, blockchain_data)
        if isinstance(result, dict) and result.get("status") == "duplicate":
            return {
                "status": "duplicate",
                "message": result["message"],
                "file_hash": result["file_hash"],
                "transaction_hash": "0x" + "0" * 62 + "duplicate"  # Mock hash indicating duplicate
            }
        return result
    except Exception:
        return "0x" + "0" * 62 + "local"  # Mock hash for local fallback

# ------------------------------------
# 🔍 Check transaction status
# ------------------------------------
def check_transaction_status(tx_hash):
    """Check the status of a transaction"""
    try:
        if not BLOCKCHAIN_ENABLED or not web3:
            return None
            
        # Remove any prefix that might indicate local storage
        if tx_hash.startswith("0x") and len(tx_hash) > 66:  # 0x + 64 hex chars
            clean_tx_hash = tx_hash[:66]  # 0x + 64 chars
        else:
            clean_tx_hash = tx_hash
            
        # Check if it's a valid transaction hash
        if not clean_tx_hash.startswith("0x") or len(clean_tx_hash) != 66:
            return None
            
        tx_receipt = web3.eth.get_transaction_receipt(clean_tx_hash)
        if tx_receipt is None:
            return {"status": "pending", "message": "Transaction is pending"}
        elif tx_receipt.status == 1:
            return {"status": "success", "message": "Transaction successful"}
        else:
            return {"status": "failed", "message": "Transaction failed"}
    except Exception as e:
        return {"status": "error", "message": f"Could not check transaction: {str(e)}"}