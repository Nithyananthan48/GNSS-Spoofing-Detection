"""
Create a demo user for development/testing.
Run this script once after starting the database:
  python create_demo_user.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.database.db import init_db, SessionLocal
from app.database.models import User
from app.utils.security import hash_password

def create_demo_user():
    init_db()
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == "demo@gnss.edu").first()
        if existing:
            print("Demo user already exists: demo@gnss.edu / demo1234")
            return
        
        user = User(
            name="Demo User",
            email="demo@gnss.edu",
            password_hash=hash_password("demo1234"),
        )
        db.add(user)
        db.commit()
        print("[SUCCESS] Demo user created!")
        print("   Email:    demo@gnss.edu")
        print("   Password: demo1234")
    finally:
        db.close()

if __name__ == "__main__":
    create_demo_user()
