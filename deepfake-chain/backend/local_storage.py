import json
import os
from datetime import datetime

# Local storage for blockchain data simulation
LOCAL_STORAGE_FILE = "blockchain_data.json"

def save_local_record(record_data):
    """Save record to local storage as a fallback when blockchain fails"""
    try:
        # Load existing data
        if os.path.exists(LOCAL_STORAGE_FILE):
            with open(LOCAL_STORAGE_FILE, 'r') as f:
                data = json.load(f)
        else:
            data = []
        
        # Add new record
        record_data["id"] = len(data) + 1
        record_data["timestamp"] = int(datetime.now().timestamp())
        data.append(record_data)
        
        # Save back to file
        with open(LOCAL_STORAGE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"💾 Saved record to local storage (ID: {record_data['id']})")
        return record_data["id"]
    except Exception as e:
        print(f"❌ Error saving to local storage: {e}")
        return None

def get_local_records():
    """Retrieve all records from local storage"""
    try:
        if os.path.exists(LOCAL_STORAGE_FILE):
            with open(LOCAL_STORAGE_FILE, 'r') as f:
                data = json.load(f)
            print(f"📂 Retrieved {len(data)} records from local storage")
            return data
        else:
            print("📭 No local storage file found")
            return []
    except Exception as e:
        print(f"❌ Error reading from local storage: {e}")
        return []

def clear_local_storage():
    """Clear all local storage data"""
    try:
        if os.path.exists(LOCAL_STORAGE_FILE):
            os.remove(LOCAL_STORAGE_FILE)
            print("🗑️ Local storage cleared")
        else:
            print("📭 No local storage file to clear")
    except Exception as e:
        print(f"❌ Error clearing local storage: {e}")

# Enhanced blockchain functions with local storage fallback
def upload_with_local_fallback(file_data, blockchain_data=None):
    """Upload to blockchain with local storage fallback"""
    from blockchain_connect import upload_to_blockchain
    
    # Save to local storage first
    if blockchain_data:
        local_id = save_local_record(blockchain_data)
        if local_id:
            print(f"📋 Local record ID: {local_id}")
    
    # Try blockchain upload
    try:
        result = upload_to_blockchain(file_data, blockchain_data)
        # Check if result is a duplicate detection response
        if isinstance(result, dict) and result.get("status") == "duplicate":
            print(f"⚠️ {result['message']}")
            # Return a special indicator for duplicates
            return {
                "status": "duplicate",
                "message": result["message"],
                "file_hash": result["file_hash"],
                "transaction_hash": "0x" + "0" * 62 + "duplicate"  # Mock hash indicating duplicate
            }
        return result
    except Exception as e:
        print(f"⚠️ Blockchain upload failed, using local storage only: {e}")
        return "0x" + "0" * 62 + "local"  # Mock hash indicating local storage

def view_with_local_fallback():
    """View blockchain data with local storage fallback"""
    from blockchain_connect import view_blockchain
    
    # Try blockchain first
    try:
        blockchain_data = view_blockchain()
        if blockchain_data and len(blockchain_data) > 0:
            print("🌐 Returning blockchain data")
            return blockchain_data
        else:
            print("📭 No blockchain data, checking local storage")
    except Exception as e:
        print(f"⚠️ Blockchain view failed: {e}")
    
    # Fallback to local storage
    local_data = get_local_records()
    if local_data:
        print("📂 Returning local storage data")
        # Format local data to match blockchain data structure
        formatted_data = []
        for record in local_data:
            formatted_data.append({
                "fileHash": record.get("hash", record.get("file_hash", "")),
                "isDeepfake": record.get("is_deepfake", False),
                "confidenceScore": record.get("confidence", 0.0),
                "uploader": record.get("uploader", "0x0000000000000000000000000000000000000000"),
                "timestamp": record.get("timestamp", 0)
            })
        return formatted_data
    
    print("📭 No data available from blockchain or local storage")
    return []