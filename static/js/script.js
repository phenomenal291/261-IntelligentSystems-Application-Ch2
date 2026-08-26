/**
 * Traffic Sign KNN Recognition Client Script
 * Ultra-Fast Asynchronous Inference with Live Parameter Controls
 */

document.addEventListener('DOMContentLoaded', () => {
    let currentImageFile = null;
    let currentImageSrc = null;
    let isProcessing = false;
    let debounceTimer = null;

    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const dropContent = document.getElementById('dropContent');
    const previewWrapper = document.getElementById('previewWrapper');
    const imagePreview = document.getElementById('imagePreview');
    const scanBar = document.getElementById('scanBar');
    const btnClear = document.getElementById('btnClear');
    const btnClassify = document.getElementById('btnClassify');
    const samplesList = document.getElementById('samplesList');
    const toast = document.getElementById('toast');

    // Hyperparameter controls
    const kSlider = document.getElementById('kSlider');
    const kBadge = document.getElementById('kBadge');
    const metricSelect = document.getElementById('metricSelect');
    const weightsSelect = document.getElementById('weightsSelect');

    // Result elements
    const resultPlaceholder = document.getElementById('resultPlaceholder');
    const resultCard = document.getElementById('resultCard');
    const resName = document.getElementById('resName');
    const resConfidence = document.getElementById('resConfidence');
    const resTime = document.getElementById('resTime');
    const origThumb = document.getElementById('origThumb');
    const hogThumb = document.getElementById('hogThumb');
    const neighborsTitle = document.getElementById('neighborsTitle');
    const activeParamsTag = document.getElementById('activeParamsTag');
    const neighborsGrid = document.getElementById('neighborsGrid');

    function showToast(message) {
        toast.textContent = message;
        toast.classList.remove('hidden');
        setTimeout(() => {
            toast.classList.add('hidden');
        }, 1800);
    }

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
            currentImageSrc = URL.createObjectURL(blob);
            displayPreview(currentImageSrc);
            showToast(`Loaded: ${sample.name}`);
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

    // 4. Clipboard Paste Support (Ctrl+V / Cmd+V)
    window.addEventListener('paste', (e) => {
        const items = (e.clipboardData || window.clipboardData).items;
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                const blob = items[i].getAsFile();
                const file = new File([blob], "pasted_image.png", { type: blob.type });
                handleFile(file);
                showToast("Image pasted from clipboard!");
                break;
            }
        }
    });

    // 5. Drag & Drop Handlers
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
            currentImageSrc = e.target.result;
            displayPreview(currentImageSrc);
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

    // 6. Clear Button
    btnClear.addEventListener('click', (e) => {
        e.stopPropagation();
        clearInput();
    });

    function clearInput() {
        currentImageFile = null;
        currentImageSrc = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewWrapper.classList.add('hidden');
        dropContent.classList.remove('hidden');
        btnClassify.disabled = true;
        scanBar.classList.add('hidden');
        resultCard.classList.add('hidden');
        resultPlaceholder.classList.remove('hidden');
    }

    // 7. Instant Debounced Parameter Re-Evaluation
    function triggerDebouncedPrediction() {
        if (!currentImageFile) return;
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            runPrediction();
        }, 50);
    }

    kSlider.addEventListener('input', (e) => {
        kBadge.textContent = e.target.value;
        triggerDebouncedPrediction();
    });

    metricSelect.addEventListener('change', triggerDebouncedPrediction);
    weightsSelect.addEventListener('change', triggerDebouncedPrediction);

    // 8. Run Instant Classification
    btnClassify.addEventListener('click', runPrediction);

    async function runPrediction() {
        if (!currentImageFile || isProcessing) return;

        isProcessing = true;
        btnClassify.disabled = true;
        btnClassify.textContent = 'Classifying...';
        scanBar.classList.remove('hidden');

        const k = parseInt(kSlider.value) || 3;
        const metric = metricSelect.value;
        const weights = weightsSelect.value;

        const formData = new FormData();
        formData.append('image', currentImageFile);
        formData.append('k', k);
        formData.append('metric', metric);
        formData.append('weights', weights);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                renderResults(data);
            } else {
                alert('Prediction error: ' + data.error);
                resultPlaceholder.classList.remove('hidden');
            }
        } catch (err) {
            console.error('Inference error:', err);
            alert('Failed to connect to server.');
            resultPlaceholder.classList.remove('hidden');
        } finally {
            scanBar.classList.add('hidden');
            btnClassify.disabled = false;
            btnClassify.innerHTML = '<span>⚡ Run KNN Classification</span>';
            isProcessing = false;
        }
    }

    // 9. Render Results & Dynamic Top-K Neighbor Gallery
    function renderResults(data) {
        resultPlaceholder.classList.add('hidden');
        resultCard.classList.remove('hidden');

        // Main stats
        resName.textContent = data.predicted_class;
        resConfidence.textContent = `${data.confidence}%`;
        resTime.textContent = `${data.inference_time_ms} ms`;

        // Active Parameters Tag
        const metricName = data.metric.charAt(0).toUpperCase() + data.metric.slice(1);
        const weightName = data.weights === 'distance' ? 'Weighted' : 'Uniform';
        activeParamsTag.textContent = `K=${data.k} • ${metricName} • ${weightName}`;
        neighborsTitle.textContent = `Top ${data.k} Nearest Neighbors in Feature Space`;

        // HOG Feature Map Visualizer
        origThumb.src = currentImageSrc;
        if (data.hog_image_b64) {
            hogThumb.src = `data:image/png;base64,${data.hog_image_b64}`;
        }

        // Render Dynamic Neighbors Gallery
        neighborsGrid.innerHTML = '';
        if (data.neighbors && data.neighbors.length > 0) {
            data.neighbors.forEach(n => {
                const card = document.createElement('div');
                card.className = 'neighbor-card';

                const imgSrc = n.image_b64 ? `data:image/png;base64,${n.image_b64}` : '/static/img/placeholder.png';
                
                card.innerHTML = `
                    <div class="neighbor-rank">#${n.rank}</div>
                    <img src="${imgSrc}" class="neighbor-img" alt="${n.class_name}">
                    <div class="neighbor-name" title="${n.class_name}">${n.class_name}</div>
                    <div class="neighbor-dist">d = ${n.distance.toFixed(3)}</div>
                    <div class="neighbor-dist">w = ${n.weight}</div>
                `;
                neighborsGrid.appendChild(card);
            });
        }
    }

    // Start
    loadSamples();
});
