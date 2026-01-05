import os
import zipfile
from huggingface_hub import hf_hub_download

def download_database():
    """Download ChromaDB from Hugging Face if not exists"""
    db_dir = os.getenv('CHROMA_DB_DIR', './chroma_db')
    
    # Check if database already exists
    if os.path.exists(db_dir) and os.path.isdir(db_dir):
        files_in_dir = os.listdir(db_dir)
        if len(files_in_dir) > 0:
            print(f"✅ Database already exists at {db_dir} with {len(files_in_dir)} items")
            return
    
    print("📥 Downloading database from Hugging Face...")
    
    try:
        # Download from Hugging Face
        zip_path = hf_hub_download(
            repo_id="sanjiv2311/solomind-database",
            filename="chroma_db.zip",
            repo_type="dataset"
        )
        
        print(f"✅ Downloaded to {zip_path}")
        
        # Extract
        print("📦 Extracting database...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        
        print(f"✅ Database extracted to {db_dir}")
        
    except Exception as e:
        print(f"❌ Error downloading database: {e}")
        print("The application will continue but queries will fail.")
        print("Please check your Hugging Face dataset is public and accessible.")

if __name__ == "__main__":
    download_database()
