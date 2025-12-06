import os
import sys
import cv2
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, url_for, send_from_directory
from werkzeug.utils import secure_filename

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.unet_plus_plus import create_model
from src.datasets.base_dataset import get_valid_transform
import albumentations as A

app = Flask(__name__, static_folder='static', static_url_path='/static')
# Use absolute path for upload folder
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# TODO: Add file validation and virus scanning

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model()
model = model.to(device)

# Load best model from organized structure
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get parent directory of webapp
model_path = os.path.join(base_dir, 'models', 'stage4_final_model.pth')

if not os.path.exists(model_path):
    # Fallback to other available models
    model_candidates = [
        os.path.join(base_dir, 'models', 'stage3_extended_model.pth'),
        os.path.join(base_dir, 'models', 'stage2_patch_finetuned.pth'), 
        os.path.join(base_dir, 'models', 'stage1_base_model.pth'),
        os.path.join(base_dir, 'sequential_finetuned_models', 'best_model_sequential_hrf_extended_drive.pth'),
        os.path.join(base_dir, 'best_model_sequential_hrf_extended_drive.pth')
    ]
    
    for candidate in model_candidates:
        if os.path.exists(candidate):
            model_path = candidate
            break
    else:
        raise FileNotFoundError("No trained model found! Please train a model first.")

print(f"Loading model from: {model_path}")
checkpoint = torch.load(model_path, map_location=device)

# Handle different checkpoint formats
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    best_dice = checkpoint.get('best_dice', 'Unknown')
else:
    model.load_state_dict(checkpoint)
    best_dice = 'Unknown'

model.eval()
print(f"Model loaded successfully! Best Dice Score: {best_dice}")

# Project information - Updated for final report
project_info = {
    'title': 'Advanced Retinal Vessel Analysis using Deep Learning for High-Resolution Image Segmentation',
    'school': 'Ho Chi Minh City University of Technology and Education',
    'faculty': 'Faculty of International Education',
    'year': '2024-2025',
    'project_code': 'CNTT2025-150',
    'project_leader': {
        'name': 'Nguyễn Nhật Phát',
        'role': 'Project Leader',
        'student_id': '23110053',
        'class': '23110FIE1'
    },
    'team_members': [
        {
            'name': 'Huỳnh Gia Hân',
            'role': 'Team Member',
            'student_id': '23110019',
            'class': '23110FIE1'
        },
        {
            'name': 'Bùi Trần Tấn Phát',
            'role': 'Team Member', 
            'student_id': '23110052',
            'class': '23110FIE1'
        },
        {
            'name': 'Trần Huỳnh Xuân Thanh',
            'role': 'Team Member',
            'student_id': '23110060',
            'class': '23110FIE1'
        }
    ],
    'supervisor': {
        'name': 'PGS.TS. Hoàng Văn Dũng',
        'role': 'Project Supervisor',
        'title': 'Associate Professor, Doctor'
    },
    'model_info': {
        'architecture': 'U-Net++ with EfficientNet-B4 backbone',
        'dataset': 'DRIVE and HRF (Sequential 4-Stage Training)',
        'performance': f'Best Dice Score: {best_dice}',
        'model_file': os.path.basename(model_path),
        'training_stages': [
            'Stage 1: Base training on DRIVE dataset',
            'Stage 2: Patch fine-tuning on HRF (512×512 patches)',
            'Stage 3: Extended training for enhanced performance',
            'Stage 4: Domain adaptation back to DRIVE'
        ]
    },
    'project_summary': {
        'description': 'Dự án này phát triển hệ thống phân đoạn mạch máu võng mạc tự động sử dụng kiến trúc U-Net++ kết hợp với EfficientNet-B4 backbone.',
        'objectives': [
            'Implement high-accuracy retinal vessel segmentation',
            'Develop multi-dataset training methodology',
            'Create production-ready web application',
            'Achieve superior performance on standard benchmarks'
        ],
        'achievements': [
            'Successfully implemented U-Net++ with EfficientNet-B4',
            'Developed 4-stage sequential training approach',
            'Achieved competitive performance on DRIVE and HRF datasets',
            'Created user-friendly web interface for real-time inference'
        ]
    }
}

def process_image(image):
    # Resize and prepare image  
    img_transform = get_valid_transform()
    
    # Ensure image is in RGB format before processing
    if len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:  # Color image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    transformed = img_transform(image=image)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        pred = (output.cpu().numpy() > 0.5).astype(np.uint8)
        pred = pred[0, 0]
    
    # Resize to original size and convert to binary image
    pred = cv2.resize(pred, (image.shape[1], image.shape[0]))
    pred = (pred > 0.5).astype(np.uint8) * 255
    
    return pred

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def home():
    return render_template('index.html', project_info=project_info)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        # Save temporary uploaded file
        filename = secure_filename(file.filename)
        temp_input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_' + filename)
        file.save(temp_input_path)
        
        try:
            # Read image and check format
            if filename.lower().endswith('.tif'):
                # Read TIF image with original format
                image = cv2.imread(temp_input_path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    return jsonify({'error': 'Cannot read TIF file'}), 400
                    
                # Normalize image if bit depth > 8
                if image.dtype != np.uint8:
                    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                # Read common image formats
                image = cv2.imread(temp_input_path)
                if image is None:
                    return jsonify({'error': 'Cannot read image file'}), 400
            
            # Get original image dimensions
            original_height, original_width = image.shape[:2]
            
            # Perform prediction
            prediction = process_image(image)
            
            # Save input image as PNG for web display
            input_filename = 'input_' + os.path.splitext(filename)[0] + '.png'
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
            
            # Process color for input image before saving
            if len(image.shape) == 2:  # Grayscale image
                # Save grayscale image directly
                cv2.imwrite(input_path, image)
            else:  # Color image
                # Save image with original colors
                cv2.imwrite(input_path, image)
            
            # Save result as PNG
            output_filename = 'output_' + os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            cv2.imwrite(output_path, prediction)
            
            # Remove temporary file
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            
            # Get processed image dimensions
            processed_height, processed_width = prediction.shape[:2]
            
            return jsonify({
                'input': url_for('static', filename='uploads/' + input_filename),
                'output': url_for('static', filename='uploads/' + output_filename),
                'input_dimensions': f'{original_width}x{original_height}',
                'output_dimensions': f'{processed_width}x{processed_height}'
            })
            
        except Exception as e:
            # Remove temporary file if error occurs
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'An error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True) 