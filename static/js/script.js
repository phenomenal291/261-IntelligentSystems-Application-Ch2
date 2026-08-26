/**
 * Traffic Sign KNN Recognition Client Script
 */

document.addEventListener('DOMContentLoaded', () => {
    let currentImageFile = null;

    // Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const dropContent = document.getElementById('dropContent');
    const previewWrapper = document.getElementById('previewWrapper');
    const imagePreview = document.getElementById('imagePreview');
    const btnClear = document.getElementById('btnClear');
    const btnClassify = document.getElementById('btnClassify');
    const samplesList = document.getElementById('samplesList');

    // Result elements
    const resultPlaceholder = document.getElementById('resultPlaceholder');
    const resultCard = document.getElementById('resultCard');
    const resName = document.getElementById('resName');
    const resConfidence = document.getElementById('resConfidence');
    const resDist = document.getElementById('resDist');
    const resTime = document.getElementById('resTime');

    // 1. Fetch & Render Demo Sample Chips
    async function loadSamples() {
        try {
            const res = await fetch('/api/samples');
            const data = await res.json();
            samplesList.innerHTML = '';
            
            data.samples.forEach(sample => {
                const chip = document.createElement('button');
                chip.type = 'button';
                chip.className = 'sample-chip';
                chip.textContent = sample.name;
                chip.addEventListener('click', () => selectSample(sample));
                samplesList.appendChild(chip);
            });
        } catch (err) {
            console.error('Error loading samples:', err);
        }
    }

    // 2. Select Sample Image
    async function selectSample(sample) {
        try {
            const res = await fetch(sample.url);
            const blob = await res.blob();
            currentImageFile = new File([blob], sample.filename, { type: 'image/png' });
            displayPreview(URL.createObjectURL(blob));
            runPrediction();
        } catch (err) {
            console.error('Error selecting sample:', err);
        }
    }

    // 3. File Input Change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    // 4. Drag & Drop Handlers
    ['dragenter', 'dragover'].forEach(name => {
        dropZone.addEventListener(name, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(name => {
        dropZone.addEventListener(name, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (PNG/JPG).');
            return;
        }
        currentImageFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            displayPreview(e.target.result);
            runPrediction();
        };
        reader.readAsDataURL(file);
    }

    function displayPreview(src) {
        imagePreview.src = src;
        dropContent.classList.add('hidden');
        previewWrapper.classList.remove('hidden');
        btnClassify.disabled = false;
    }

    // 5. Clear Button
    btnClear.addEventListener('click', (e) => {
        e.stopPropagation();
        clearInput();
    });

    function clearInput() {
        currentImageFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewWrapper.classList.add('hidden');
        dropContent.classList.remove('hidden');
        btnClassify.disabled = true;
        resultCard.classList.add('hidden');
        resultPlaceholder.classList.remove('hidden');
    }

    // 6. Run Prediction via Fetch API
    btnClassify.addEventListener('click', runPrediction);

    async function runPrediction() {
        if (!currentImageFile) return;

        btnClassify.disabled = true;
        btnClassify.textContent = 'Processing...';

        const formData = new FormData();
        formData.append('image', currentImageFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (data.success) {
                resultPlaceholder.classList.add('hidden');
                resultCard.classList.remove('hidden');

                resName.textContent = data.predicted_class;
                resConfidence.textContent = `${data.confidence}%`;
                resDist.textContent = data.min_neighbor_distance.toFixed(4);
                resTime.textContent = `${data.inference_time_ms} ms`;
            } else {
                alert('Prediction error: ' + data.error);
            }
        } catch (err) {
            console.error('Inference error:', err);
            alert('Failed to connect to server.');
        } finally {
            btnClassify.disabled = false;
            btnClassify.textContent = 'Run KNN Prediction';
        }
    }

    // Initialize
    loadSamples();
});
