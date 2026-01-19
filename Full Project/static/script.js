function openTab(tabName) {
    const contents = document.querySelectorAll('.tab-content');
    const buttons = document.querySelectorAll('.tab-btn');
    
    contents.forEach(content => content.classList.remove('active'));
    buttons.forEach(btn => btn.classList.remove('active'));
    
    document.getElementById(tabName).classList.add('active');
    event.currentTarget.classList.add('active');
}

// Manual Prediction
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = e.target.querySelector('button');
    const originalText = btn.innerHTML;
    btn.innerHTML = 'Calculating...';
    btn.disabled = true;

    try {
        const formData = new FormData(e.target);
        const data = Object.fromEntries(formData.entries());

        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        
        if (response.ok) {
            document.getElementById('priceValue').textContent = result.prediction;
            document.getElementById('resultArea').classList.remove('hidden');
            // Scroll to result
            document.getElementById('resultArea').scrollIntoView({ behavior: 'smooth' });
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        alert('An error occurred. Please check the console.');
        console.error(error);
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
});

// Batch Upload
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelect();
    }
});

fileInput.addEventListener('change', handleFileSelect);

function handleFileSelect() {
    if (fileInput.files.length) {
        dropZone.innerHTML = `<p>Selected: <strong>${fileInput.files[0].name}</strong></p>`;
        uploadBtn.classList.remove('hidden');
    }
}

uploadBtn.addEventListener('click', async () => {
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    uploadBtn.innerHTML = 'Processing...';
    uploadBtn.disabled = true;

    try {
        const response = await fetch('/predict_batch', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'predictions.csv';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            uploadBtn.innerHTML = 'Download Complete!';
        } else {
            const err = await response.json();
            alert('Error: ' + err.error);
            uploadBtn.innerHTML = 'Try Again';
        }
    } catch (error) {
        console.error(error);
        alert('Upload failed.');
        uploadBtn.innerHTML = 'Try Again';
    } finally {
        uploadBtn.disabled = false;
    }
});
