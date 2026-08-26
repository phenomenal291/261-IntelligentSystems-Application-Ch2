/**
 * Traffic Sign KNN Recognition Client Script
 * Includes Clipboard Paste Support & Educational Pipeline Visualizer
 */

document.addEventListener('DOMContentLoaded', () => {
    let currentImageFile = null;
    let currentImageSrc = null;

    // Elements
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

    // Stepper elements
    const pipelineStepper = document.getElementById('pipelineStepper');
    const step1 = document.getElementById('step1');
    const step2 = document.getElementById('step2');
    const step3 = document.getElementById('step3');
    const step4 = document.getElementById('step4');
    const line1 = document.getElementById('line1');
    const line2 = document.getElementById('line2');
    const line3 = document.getElementById('line3');

    // Result elements
    const resultPlaceholder = document.getElementById('resultPlaceholder');
    const resultCard = document.getElementById('resultCard');
    const resName = document.getElementById('resName');
    const resConfidence = document.getElementById('resConfidence');
    const resTime = document.getElementById('resTime');
    const origThumb = document.getElementById('origThumb');
    const hogThumb = document.getElementById('hogThumb');
    const neighborsGrid = document.getElementById('neighborsGrid');

    function showToast(message) {
        toast.textContent = message;
        toast.classList.remove('hidden');
        setTimeout(() => {
            toast.classList.add('hidden');
        }, 2500);
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

    // 2. Select Sample Image (Loads cleanly into preview)
    async function selectSample(sample) {
        try {
            const res = await fetch(sample.url);
            const blob = await res.blob();
            currentImageFile = new File([blob], sample.filename, { type: 'image/png' });
            currentImageSrc = URL.createObjectURL(blob);
            displayPreview(currentImageSrc);
            showToast(`Loaded sample: ${sample.name}`);
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
        };
        reader.readAsDataURL(file);
    }

    function displayPreview(src) {
        imagePreview.src = src;
        dropContent.classList.add('hidden');
        previewWrapper.classList.remove('hidden');
        btnClassify.disabled = false;
        btnClassify.innerHTML = '<span>⚡ Run KNN Recognition Pipeline</span>';
        
        // Reset results on new image
        resultCard.classList.add('hidden');
        resultPlaceholder.classList.remove('hidden');
        pipelineStepper.classList.add('hidden');
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
        pipelineStepper.classList.add('hidden');
    }

    // 7. Run Pipeline with Educational Step-by-Step Animation
    btnClassify.addEventListener('click', runPrediction);

    function setStep(stepNum) {
        [step1, step2, step3, step4].forEach((s, idx) => {
            s.classList.remove('active', 'done');
            if (idx + 1 < stepNum) s.classList.add('done');
            else if (idx + 1 === stepNum) s.classList.add('active');
        });
        [line1, line2, line3].forEach((l, idx) => {
            if (idx + 1 < stepNum) l.classList.add('active');
            else l.classList.remove('active');
        });
    }

    async function runPrediction() {
        if (!currentImageFile) return;

        btnClassify.disabled = true;
        btnClassify.textContent = 'Processing Pipeline...';
        scanBar.classList.remove('hidden');
        pipelineStepper.classList.remove('hidden');
        resultPlaceholder.classList.add('hidden');
        resultCard.classList.add('hidden');

        // Step 1: Preprocessing
        setStep(1);
        await new Promise(r => setTimeout(r, 250));

        // Step 2: HOG extraction
        setStep(2);
        await new Promise(r => setTimeout(r, 300));

        // Step 3: Compute distances
        setStep(3);

        const formData = new FormData();
        formData.append('image', currentImageFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            // Step 4: Majority Voting
            setStep(4);
            await new Promise(r => setTimeout(r, 250));

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
            btnClassify.innerHTML = '<span>⚡ Run KNN Recognition Pipeline</span>';
        }
    }

    // 8. Render Rich Educational Results
    function renderResults(data) {
        resultPlaceholder.classList.add('hidden');
        resultCard.classList.remove('hidden');

        // Main stats
        resName.textContent = data.predicted_class;
        resConfidence.textContent = `${data.confidence}%`;
        resTime.textContent = `${data.inference_time_ms} ms`;

        // HOG Feature Map Visualizer
        origThumb.src = currentImageSrc;
        if (data.hog_image_b64) {
            hogThumb.src = `data:image/png;base64,${data.hog_image_b64}`;
        }

        // K=3 Nearest Neighbors Gallery
        neighborsGrid.innerHTML = '';
        if (data.neighbors && data.neighbors.length > 0) {
            data.neighbors.forEach(n => {
                const card = document.createElement('div');
                card.className = 'neighbor-card';

                const imgSrc = n.image_b64 ? `data:image/png;base64,${n.image_b64}` : '/static/img/placeholder.png';
                
                card.innerHTML = `
                    <div class="neighbor-rank">Rank #${n.rank}</div>
                    <img src="${imgSrc}" class="neighbor-img" alt="${n.class_name}">
                    <div class="neighbor-name">${n.class_name}</div>
                    <div class="neighbor-dist">d = ${n.distance.toFixed(4)}</div>
                    <div class="neighbor-dist">w = ${n.weight}</div>
                `;
                neighborsGrid.appendChild(card);
            });
        }
    }

    // Start
    loadSamples();
});
