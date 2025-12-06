document.addEventListener('DOMContentLoaded', function() {
    const uploadBox = document.getElementById('upload-box');
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const loadingSpinner = document.getElementById('loading');
    const resultsSection = document.getElementById('results-section');
    const inputImage = document.getElementById('input-image');
    const outputImage = document.getElementById('output-image');
    const errorMessage = document.getElementById('error-message');
    
    // Handle click on upload box
    uploadBox.addEventListener('click', () => {
        fileInput.click();
    });
    
    // Handle file selection
    fileInput.addEventListener('change', () => {
        if (fileInput.files && fileInput.files[0]) {
            const file = fileInput.files[0];
            
            // Check file format
            if (!file.type.match('image.*')) {
                showError('Please select an image file');
                return;
            }
            
            // Display selected file name
            document.getElementById('file-name').textContent = file.name;
            uploadBtn.style.display = 'inline-block';
        }
    });
    
    // Handle file upload
    uploadBtn.addEventListener('click', async () => {
        if (!fileInput.files || !fileInput.files[0]) {
            showError('Please select an image file');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        try {
            // Show loading
            loadingSpinner.style.display = 'block';
            uploadBtn.style.display = 'none';
            errorMessage.style.display = 'none';
            
            // Send request
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Display results
                inputImage.src = data.input;
                outputImage.src = data.output;
                
                // Display image dimensions
                document.getElementById('input-dimensions').textContent = `Dimensions: ${data.input_dimensions} pixels`;
                document.getElementById('output-dimensions').textContent = `Dimensions: ${data.output_dimensions} pixels`;
                
                resultsSection.style.display = 'block';
            } else {
                showError(data.error || 'An error occurred');
            }
        } catch (error) {
            showError('An error occurred while processing the image');
        } finally {
            loadingSpinner.style.display = 'none';
            uploadBtn.style.display = 'inline-block';
        }
    });
    
    // Function to display error
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
    }
    
    // Drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadBox.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadBox.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadBox.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        uploadBox.classList.add('highlight');
    }
    
    function unhighlight() {
        uploadBox.classList.remove('highlight');
    }
    
    uploadBox.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files && files[0]) {
            fileInput.files = files;
            document.getElementById('file-name').textContent = files[0].name;
            uploadBtn.style.display = 'inline-block';
        }
    }
}); 