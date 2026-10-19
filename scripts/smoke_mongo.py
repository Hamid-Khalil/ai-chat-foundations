import asyncio
import os
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import AsyncMongoClient, ReturnDocument, ASCENDING
from pymongo.errors import DuplicateKeyError, OperationFailure, ServerSelectionTimeoutError

load_dotenv()

URI = os.getenv("MONGODB_URI")
if not URI:
    raise ValueError("MONGODB_URI not found in .env")

async def run_smoke_test():
    print("=== Task 0.3: PyMongo Async Smoke Test ===\n")
    
    # 1. Connect and ping
    client = AsyncMongoClient(URI, serverSelectionTimeoutMS=3000)
    db = client.get_default_database()
    smoke = db["smoke"]
    await smoke.drop()  # Clean slate
    
    print("1. Ping Database:")
    ping_res = await client.admin.command("ping")
    print(f"   Ping result: {ping_res}")
    
    # 2. Datetimes and tz_aware
    print("\n2. Datetimes and tz_aware:")
    doc_id = "doc_tz_1"
    now_utc = datetime.now(timezone.utc)
    await smoke.insert_one({"_id": doc_id, "created_at": now_utc})
    
    doc_naive = await smoke.find_one({"_id": doc_id})
    print(f"   Naive Client - _id type: {type(doc_naive['_id']).__name__}")
    print(f"   Naive Client - tzinfo: {doc_naive['created_at'].tzinfo}")
    
    client_aware = AsyncMongoClient(URI, serverSelectionTimeoutMS=3000, tz_aware=True)
    db_aware = client_aware.get_default_database()
    doc_aware = await db_aware.smoke.find_one({"_id": doc_id})
    print(f"   Aware Client - tzinfo: {doc_aware['created_at'].tzinfo}")
    await client_aware.close()
    
    # 3. Missing ID
    print("\n3. Missing ID:")
    missing_doc = await smoke.find_one({"_id": "does_not_exist"})
    print(f"   Result for missing ID: {missing_doc}")
    
    # 4. Duplicate _id
    print("\n4. Duplicate _id exception:")
    try:
        await smoke.insert_one({"_id": doc_id, "created_at": now_utc})
    except DuplicateKeyError as e:
        print(f"   Caught expected exception: {type(e).__name__}")
    
    # 5. Unique Index
    print("\n5. Unique Index:")
    await smoke.create_index([("my_unique_field", ASCENDING)], unique=True)
    await smoke.insert_one({"_id": "u1", "my_unique_field": "apple"})
    try:
        await smoke.insert_one({"_id": "u2", "my_unique_field": "apple"})
    except DuplicateKeyError as e:
        print(f"   Caught duplicate index exception: {type(e).__name__}")
    
    idx_info = await smoke.index_information()
    print(f"   Index information keys: {list(idx_info.keys())}")
    
    # Run create_index again (idempotent)
    await smoke.create_index([("my_unique_field", ASCENDING)], unique=True)
    print("   Running create_index twice is safe.")
    
    # Drop the index so subsequent documents aren't constrained by null values
    await smoke.drop_index("my_unique_field_1")
    
    # 6. Pagination & to_list
    print("\n6. Pagination (sort, limit, to_list):")
    await smoke.insert_many([{"_id": f"p{i}", "val": i} for i in range(5)])
    items = await smoke.find({"_id": {"$regex": "^p"}}).sort("val", ASCENDING).limit(3).to_list(None)
    print(f"   to_list(None) fetched: {[item['val'] for item in items]}")
    try:
        await smoke.find({"_id": {"$regex": "^p"}}).sort("val", ASCENDING).limit(3).to_list(0)
    except Exception as e:
        print(f"   to_list(0) raised: {type(e).__name__} ({e})")
        
    # 7. Atomic Counter vs Naive Counter
    print("\n7. Atomic Counter vs Naive Counter (20 concurrent requests):")
    await smoke.insert_one({"_id": "c", "n": 0})
    await smoke.insert_one({"_id": "naive_counter", "n": 0})
    
    async def atomic_increment():
        res = await smoke.find_one_and_update(
            {"_id": "c"}, 
            {"$inc": {"n": 1}}, 
            return_document=ReturnDocument.AFTER
        )
        return res["n"]
        
    async def naive_increment():
        doc = await smoke.find_one({"_id": "naive_counter"})
        new_n = doc["n"] + 1
        await asyncio.sleep(0.01)  # Simulate network/processing delay
        await smoke.update_one({"_id": "naive_counter"}, {"$set": {"n": new_n}})
        return new_n
        
    atomic_results = await asyncio.gather(*[atomic_increment() for _ in range(20)])
    naive_results = await asyncio.gather(*[naive_increment() for _ in range(20)])
    
    print(f"   Atomic Results (1-20 unique): {sorted(atomic_results)}")
    print(f"   Naive Results (lost updates!): {sorted(naive_results)}")

    # 8. Failure Mode (c): Wrong Password
    print("\n8. Failure Mode (c) - Wrong Password:")
    bad_uri = URI.replace("local_app_secret_2026", "wrong_password_123")
    bad_client = AsyncMongoClient(bad_uri, serverSelectionTimeoutMS=3000)
    start_t = time.time()
    try:
        await bad_client.admin.command("ping")
        print("   Unexpected success with bad password.")
    except OperationFailure as e:
        dur = round(time.time() - start_t, 2)
        print(f"   Caught OperationFailure (auth failed) in {dur}s")
    finally:
        await bad_client.close()
    
    # Clean up
    await smoke.drop()
    await client.close()
    print("\n=== Smoke Test Complete ===")

if __name__ == "__main__":
    asyncio.run(run_smoke_test())
