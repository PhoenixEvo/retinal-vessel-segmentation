import os
from sklearn.model_selection import train_test_split

def get_hrf_train_test_split():
    """Get the exact same train/test split used during training"""
    images_dir = 'data/HRF/images'
    
    image_files = []
    for ext in ['.jpg', '.JPG', '.png', '.tif']:
        image_files.extend([f for f in os.listdir(images_dir) if f.endswith(ext)])
    image_files = sorted(image_files)
    
    # Use SAME split as training (random_state=42, test_size=0.2)
    train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)
    
    return train_files, test_files, image_files

def check_evaluation_files():
    """Check which files were used in evaluation"""
    
    print("KIỂM TRA DATA LEAKAGE TRONG HRF DATASET")
    print("="*60)
    
    # Get training split
    train_files, test_files, all_files = get_hrf_train_test_split()
    
    print(f"\n📊 TRAINING SPLIT (random_state=42, test_size=0.2):")
    print(f"   - Tổng số images: {len(all_files)}")
    print(f"   - Train images: {len(train_files)} (80%)")
    print(f"   - Test images: {len(test_files)} (20%)")
    
    print(f"\n🚂 TRAIN SET ({len(train_files)} images):")
    for i, f in enumerate(train_files, 1):
        print(f"   {i:2d}. {f}")
    
    print(f"\n🧪 TEST SET ({len(test_files)} images):")
    for i, f in enumerate(test_files, 1):
        print(f"   {i:2d}. {f}")
    
    # Check evaluation results
    print(f"\n🔍 KIỂM TRA FILES TRONG EVALUATION RESULTS:")
    
    # Check hrf_test_evaluation_results.csv
    test_eval_file = 'hrf_test_evaluation_results.csv'
    if os.path.exists(test_eval_file):
        print(f"\n📄 {test_eval_file}:")
        with open(test_eval_file, 'r') as f:
            lines = f.readlines()
            eval_files = []
            for line in lines[1:]:  # Skip header
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) > 6:
                        eval_files.append(parts[6])  # image_name column
        
        print(f"   - Số files được evaluate: {len(eval_files)}")
        for i, f in enumerate(eval_files, 1):
            print(f"   {i:2d}. {f}")
        
        # Check if these are test files only
        test_only = all(f in test_files for f in eval_files)
        print(f"\n✅ Chỉ dùng TEST SET: {'YES' if test_only else 'NO'}")
        
        if not test_only:
            print("⚠️  CẢNH BÁO: Có files từ TRAINING SET!")
            train_in_eval = [f for f in eval_files if f in train_files]
            print(f"Files từ train set: {train_in_eval}")
    
    # Check hrf_evaluation_results.csv
    all_eval_file = 'hrf_evaluation_results.csv'
    if os.path.exists(all_eval_file):
        print(f"\n📄 {all_eval_file}:")
        with open(all_eval_file, 'r') as f:
            lines = f.readlines()
            eval_files = []
            for line in lines[1:]:  # Skip header
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) > 6:
                        eval_files.append(parts[6])  # image_name column
        
        print(f"   - Số files được evaluate: {len(eval_files)}")
        
        # Check composition
        test_in_eval = [f for f in eval_files if f in test_files]
        train_in_eval = [f for f in eval_files if f in train_files]
        
        print(f"   - Files từ TEST SET: {len(test_in_eval)}")
        print(f"   - Files từ TRAIN SET: {len(train_in_eval)}")
        
        if len(train_in_eval) > 0:
            print("⚠️  CẢNH BÁO: DATA LEAKAGE DETECTED!")
            print("Files từ train set được dùng trong evaluation:")
            for f in train_in_eval:
                print(f"     - {f}")
        else:
            print("✅ Không có data leakage")

    print(f"\n📋 TÓM TẮT:")
    print(f"   - Model train trên: {len(train_files)} images (80%)")
    print(f"   - Evaluation đúng cách phải chỉ dùng: {len(test_files)} images (20%)")
    
    # Check which results file is correct
    if os.path.exists(test_eval_file):
        print(f"   - {test_eval_file}: Đánh giá đúng (chỉ test set)")
    if os.path.exists(all_eval_file):
        print(f"   - {all_eval_file}: CÓ THỂ có data leakage (cần kiểm tra)")

if __name__ == "__main__":
    check_evaluation_files()

