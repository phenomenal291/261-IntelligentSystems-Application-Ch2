/**
 * Traffic Sign KNN Recognition Client Script
 * Visualizes 2D PCA Feature Space Map, Dual Voting Comparisons, & Top-K Neighbors
 */

document.addEventListener('DOMContentLoaded', () => {
    let currentImageFile = null;
    let currentImageSrc = null;
    let isProcessing = false;
    let debounceTimer = null;

    // Feature Map Data
    let mapData = null;
    let query2D = null;
    let activeNeighbors = [];
    let canvasBounds = { xMin: -5, xMax: 5, yMin: -5, yMax: 5 };

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

    // Controls
    const kSlider = document.getElementById('kSlider');
    const kBadge = document.getElementById('kBadge');
    const metricSelect = document.getElementById('metricSelect');
    const weightsSelect = document.getElementById('weightsSelect');

    // Result Elements
    const resName = document.getElementById('resName');
    const resConfidence = document.getElementById('resConfidence');
    const resTime = document.getElementById('resTime');
    const origThumb = document.getElementById('origThumb');
    const hogThumb = document.getElementById('hogThumb');
    const featureCompWrapper = document.getElementById('featureCompWrapper');
    const featureCompPlaceholder = document.getElementById('featureCompPlaceholder');
    const mapStatusTag = document.getElementById('mapStatusTag');
    const uniformBars = document.getElementById('uniformBars');
    const distanceBars = document.getElementById('distanceBars');
    const neighborsTitle = document.getElementById('neighborsTitle');
    const neighborsGrid = document.getElementById('neighborsGrid');

    // Canvas
    const canvas = document.getElementById('featureMapCanvas');
    const ctx = canvas.getContext('2d');
    const tooltip = document.getElementById('canvasTooltip');

    function showToast(message) {
        toast.textContent = message;
        toast.classList.remove('hidden');
        setTimeout(() => toast.classList.add('hidden'), 1800);
    }

    // 1. Initialize & Fetch Static Feature Map Data
    async function init() {
        try {
            // Load sample signs
            const resSamples = await fetch('/api/samples');
            const dataSamples = await resSamples.json();
            samplesList.innerHTML = '';
            dataSamples.samples.forEach(sample => {
                const chip = document.createElement('button');
                chip.type = 'button';
                chip.className = 'sample-chip';
                chip.textContent = sample.name;
                chip.addEventListener('click', () => selectSample(sample));
                samplesList.appendChild(chip);
            });

            // Load 2D Feature Map points
            const resMap = await fetch('/api/feature_map');
            mapData = await resMap.json();
            computeCanvasBounds();
            resizeCanvas();
            drawFeatureMap();
        } catch (err) {
            console.error('Initialization error:', err);
        }
    }

    function computeCanvasBounds() {
        if (!mapData || !mapData.points.length) return;
        const xs = mapData.points.map(p => p.x);
        const ys = mapData.points.map(p => p.y);
        const padX = (Math.max(...xs) - Math.min(...xs)) * 0.12 || 1;
        const padY = (Math.max(...ys) - Math.min(...ys)) * 0.12 || 1;
        canvasBounds = {
            xMin: Math.min(...xs) - padX,
            xMax: Math.max(...xs) + padX,
            yMin: Math.min(...ys) - padY,
            yMax: Math.max(...ys) + padY
        };
    }

    function toCanvasCoords(x, y) {
        const cw = canvas.width;
        const ch = canvas.height;
        const cx = ((x - canvasBounds.xMin) / (canvasBounds.xMax - canvasBounds.xMin)) * cw;
        const cy = ch - ((y - canvasBounds.yMin) / (canvasBounds.yMax - canvasBounds.yMin)) * ch;
        return { cx, cy };
    }

    function resizeCanvas() {
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * window.devicePixelRatio || 560;
        canvas.height = rect.height * window.devicePixelRatio || 340;
    }
    window.addEventListener('resize', () => {
        resizeCanvas();
        drawFeatureMap();
    });

    // 2. Draw 2D Feature Space Map
    function drawFeatureMap() {
        if (!mapData || !ctx) return;
        const cw = canvas.width;
        const ch = canvas.height;
        ctx.clearRect(0, 0, cw, ch);

        // Draw subtle grid
        ctx.strokeStyle = '#1E293B';
        ctx.lineWidth = 1;
        for (let x = 0; x < cw; x += 40) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, ch); ctx.stroke();
        }
        for (let y = 0; y < ch; y += 40) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cw, y); ctx.stroke();
        }

        // Draw connecting rays & radius circle if Query active
        if (query2D && activeNeighbors.length > 0) {
            const qPt = toCanvasCoords(query2D[0], query2D[1]);
            
            // Max neighbor distance for radius circle
            let maxCanvasDist = 0;
            activeNeighbors.forEach(n => {
                const nPt = toCanvasCoords(n.x2d, n.y2d);
                const dPx = Math.hypot(qPt.cx - nPt.cx, qPt.cy - nPt.cy);
                if (dPx > maxCanvasDist) maxCanvasDist = dPx;

                // Connecting Ray
                ctx.beginPath();
                ctx.moveTo(qPt.cx, qPt.cy);
                ctx.lineTo(nPt.cx, nPt.cy);
                ctx.strokeStyle = n.color || '#3B82F6';
                ctx.lineWidth = 1.8;
                ctx.setLineDash([4, 4]);
                ctx.stroke();
                ctx.setLineDash([]);
            });

            // Enclosing Neighborhood Circle
            if (maxCanvasDist > 0) {
                ctx.beginPath();
                ctx.arc(qPt.cx, qPt.cy, maxCanvasDist + 8, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(59, 130, 246, 0.08)';
                ctx.fill();
                ctx.strokeStyle = 'rgba(59, 130, 246, 0.4)';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
        }

        // Draw all training cluster points
        mapData.points.forEach(p => {
            const pt = toCanvasCoords(p.x, p.y);
            const isNeighbor = activeNeighbors.some(n => n.index === p.index);

            ctx.beginPath();
            ctx.arc(pt.cx, pt.cy, isNeighbor ? 5.5 : 3.5, 0, Math.PI * 2);
            ctx.fillStyle = p.color;
            ctx.fill();

            if (isNeighbor) {
                ctx.strokeStyle = '#FFFFFF';
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        });

        // Draw Query Point Star / Badge
        if (query2D) {
            const qPt = toCanvasCoords(query2D[0], query2D[1]);

            // Outer glow pulse
            ctx.beginPath();
            ctx.arc(qPt.cx, qPt.cy, 12, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(234, 179, 8, 0.25)';
            ctx.fill();

            // Inner target
            ctx.beginPath();
            ctx.arc(qPt.cx, qPt.cy, 7, 0, Math.PI * 2);
            ctx.fillStyle = '#EAB308';
            ctx.fill();
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 2.5;
            ctx.stroke();

            // Label
            ctx.fillStyle = '#F8FAFC';
            ctx.font = 'bold 12px Inter, sans-serif';
            ctx.fillText('Query (Q)', qPt.cx + 10, qPt.cy - 6);
        }
    }

    // 3. Canvas Hover Tooltip
    canvas.addEventListener('mousemove', (e) => {
        if (!mapData) return;
        const rect = canvas.getBoundingClientRect();
        const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
        const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);

        let found = null;
        for (const p of mapData.points) {
            const pt = toCanvasCoords(p.x, p.y);
            if (Math.hypot(mouseX - pt.cx, mouseY - pt.cy) < 10) {
                found = p;
                break;
            }
        }

        if (found) {
            tooltip.classList.remove('hidden');
            tooltip.style.left = `${e.clientX - rect.left + 12}px`;
            tooltip.style.top = `${e.clientY - rect.top + 12}px`;
            tooltip.textContent = `${found.name} (x: ${found.x}, y: ${found.y})`;
        } else {
            tooltip.classList.add('hidden');
        }
    });

    canvas.addEventListener('mouseleave', () => tooltip.classList.add('hidden'));

    // 4. Sample Selection
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

    // 5. File Input Change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    // 6. Clipboard Paste Support (Ctrl+V / Cmd+V)
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

    // 7. Drag & Drop Handlers
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

    // 8. Clear Button
    btnClear.addEventListener('click', (e) => {
        e.stopPropagation();
        clearInput();
    });

    function clearInput() {
        currentImageFile = null;
        currentImageSrc = null;
        query2D = null;
        activeNeighbors = [];
        fileInput.value = '';
        imagePreview.src = '';
        previewWrapper.classList.add('hidden');
        dropContent.classList.remove('hidden');
        btnClassify.disabled = true;
        scanBar.classList.add('hidden');
        resName.textContent = 'Ready';
        resConfidence.textContent = '--%';
        resTime.textContent = '-- ms';
        uniformBars.innerHTML = '<span class="text-muted">Run classification to view votes.</span>';
        distanceBars.innerHTML = '<span class="text-muted">Run classification to view votes.</span>';
        neighborsGrid.innerHTML = '<div class="text-muted text-center py-2">Select an image to inspect nearest training exemplars.</div>';
        if (featureCompWrapper) featureCompWrapper.classList.add('hidden');
        if (featureCompPlaceholder) featureCompPlaceholder.classList.remove('hidden');
        drawFeatureMap();
    }

    // 9. Instant Parameter Re-Evaluation
    function triggerDebouncedPrediction() {
        if (!currentImageFile) return;
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            runPrediction();
        }, 50);
    }

    function updateKValue(val) {
        const kNum = parseInt(val);
        kSlider.value = kNum;
        kBadge.textContent = kNum;

        // Update active preset button styling
        document.querySelectorAll('.btn-k-preset').forEach(btn => {
            if (parseInt(btn.dataset.k) === kNum) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        triggerDebouncedPrediction();
    }

    kSlider.addEventListener('input', (e) => {
        updateKValue(e.target.value);
    });

    document.querySelectorAll('.btn-k-preset').forEach(btn => {
        btn.addEventListener('click', () => {
            updateKValue(btn.dataset.k);
        });
    });

    metricSelect.addEventListener('change', triggerDebouncedPrediction);
    weightsSelect.addEventListener('change', triggerDebouncedPrediction);

    // 10. Run Prediction
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
            }
        } catch (err) {
            console.error('Inference error:', err);
            alert('Failed to connect to server.');
        } finally {
            scanBar.classList.add('hidden');
            btnClassify.disabled = false;
            btnClassify.innerHTML = '<span>⚡ Run KNN Classification</span>';
            isProcessing = false;
        }
    }

    // 11. Render Results, Feature Map, & Voting Bars
    function renderResults(data) {
        // Main stats
        resName.textContent = data.predicted_class;
        resConfidence.textContent = `${data.confidence}%`;
        resTime.textContent = `${data.inference_time_ms} ms`;

        // Update Map State
        query2D = data.q_2d;
        activeNeighbors = data.neighbors;
        const metricName = data.metric.charAt(0).toUpperCase() + data.metric.slice(1);
        const weightName = data.weights === 'distance' ? 'Weighted' : 'Uniform';
        mapStatusTag.textContent = `K=${data.k} • ${metricName} • ${weightName}`;
        neighborsTitle.textContent = `Top ${data.k} Nearest Neighbors Gallery`;

        // Draw Map
        drawFeatureMap();

        // HOG Thumbnails
        origThumb.src = currentImageSrc;
        if (data.hog_image_b64) {
            hogThumb.src = `data:image/png;base64,${data.hog_image_b64}`;
        }
        if (featureCompWrapper) featureCompWrapper.classList.remove('hidden');
        if (featureCompPlaceholder) featureCompPlaceholder.classList.add('hidden');

        // Render Voting Comparison Bars
        renderVotingBars(uniformBars, data.voting_comparison.uniform, false);
        renderVotingBars(distanceBars, data.voting_comparison.distance, true);

        // Render Neighbors Gallery
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

    function renderVotingBars(container, list, isDistance) {
        container.innerHTML = '';
        if (!list || !list.length) {
            container.innerHTML = '<span class="text-muted">No votes.</span>';
            return;
        }

        list.forEach(item => {
            const row = document.createElement('div');
            row.className = 'vote-row';

            const scoreLabel = isDistance ? `${item.score} pts (${item.percentage}%)` : `${item.score} votes (${item.percentage}%)`;

            row.innerHTML = `
                <div class="vote-row-meta">
                    <span>${item.class_name}</span>
                    <span>${scoreLabel}</span>
                </div>
                <div class="vote-progress-track">
                    <div class="vote-progress-fill" style="width: ${item.percentage}%; background-color: ${item.color || '#3B82F6'};"></div>
                </div>
            `;
            container.appendChild(row);
        });
    }

    init();
});
