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
        setTimeout(() => toast.classList.add('hidden'), 2600);
    }

    // Attach click handlers to sample buttons by index
    function setupSampleChips() {
        document.querySelectorAll('.sample-chip').forEach(chip => {
            chip.onclick = () => {
                const idx = parseInt(chip.dataset.idx);
                if (window.PRELOADED_SAMPLES && window.PRELOADED_SAMPLES[idx]) {
                    selectSample(window.PRELOADED_SAMPLES[idx]);
                }
            };
        });
    }

    // 1. Initialize & Render Feature Space Map
    async function init() {
        setupSampleChips();

        if (window.PRELOADED_FEATURE_MAP && window.PRELOADED_FEATURE_MAP.points && window.PRELOADED_FEATURE_MAP.points.length > 0) {
            mapData = window.PRELOADED_FEATURE_MAP;
            computeCanvasBounds();
            resizeCanvas();
            drawFeatureMap();
        } else {
            try {
                let resMap = await fetch('/api/feature_map').catch(() => fetch('/feature_map'));
                if (resMap && resMap.ok) {
                    mapData = await resMap.json();
                    computeCanvasBounds();
                    resizeCanvas();
                    drawFeatureMap();
                }
            } catch (err) {
                console.error('Initialization error fetching feature map:', err);
            }
        }
    }

    function computeCanvasBounds() {
        if (!mapData || !mapData.points || !mapData.points.length) {
            canvasBounds = { xMin: -5, xMax: 5, yMin: -5, yMax: 5 };
            return;
        }
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (let i = 0; i < mapData.points.length; i++) {
            const p = mapData.points[i];
            if (p.x < minX) minX = p.x;
            if (p.x > maxX) maxX = p.x;
            if (p.y < minY) minY = p.y;
            if (p.y > maxY) maxY = p.y;
        }
        if (maxX <= minX) { minX -= 1; maxX += 1; }
        if (maxY <= minY) { minY -= 1; maxY += 1; }

        const padX = (maxX - minX) * 0.10 || 1;
        const padY = (maxY - minY) * 0.10 || 1;
        canvasBounds = {
            xMin: minX - padX,
            xMax: maxX + padX,
            yMin: minY - padY,
            yMax: maxY + padY
        };
    }

    function toCanvasCoords(x, y) {
        const cw = canvas.width || 600;
        const ch = canvas.height || 310;
        const rangeX = (canvasBounds.xMax - canvasBounds.xMin) || 1;
        const rangeY = (canvasBounds.yMax - canvasBounds.yMin) || 1;
        const cx = ((x - canvasBounds.xMin) / rangeX) * cw;
        const cy = ch - ((y - canvasBounds.yMin) / rangeY) * ch;
        return { cx, cy };
    }

    function resizeCanvas() {
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const w = (rect && rect.width > 0) ? rect.width : (canvas.parentElement ? canvas.parentElement.clientWidth : 560);
        const h = (rect && rect.height > 0) ? rect.height : 310;
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
    }

    window.addEventListener('resize', () => {
        resizeCanvas();
        drawFeatureMap();
    });

    // 2. Draw 2D Feature Space Map
    function drawFeatureMap() {
        if (!mapData || !ctx || !canvas) return;
        if (canvas.width === 0 || canvas.height === 0) {
            resizeCanvas();
        }
        const width = canvas.width || 600;
        const height = canvas.height || 310;
        const dpr = window.devicePixelRatio || 1;

        ctx.clearRect(0, 0, width, height);

        // Background
        ctx.fillStyle = '#0F172A';
        ctx.fillRect(0, 0, width, height);

        // Subtle Grid
        ctx.strokeStyle = '#1E293B';
        ctx.lineWidth = 1;
        const gridStep = Math.max(30, Math.round(40 * dpr));
        for (let x = 0; x < width; x += gridStep) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
        }
        for (let y = 0; y < height; y += gridStep) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
        }

        // Connecting rays & radius circle if Query active
        if (query2D && activeNeighbors && activeNeighbors.length > 0) {
            const qPt = toCanvasCoords(query2D[0], query2D[1]);
            
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
                ctx.lineWidth = 2 * dpr;
                ctx.setLineDash([5, 5]);
                ctx.stroke();
                ctx.setLineDash([]);
            });

            // Enclosing Neighborhood Circle
            if (maxCanvasDist > 0) {
                ctx.beginPath();
                ctx.arc(qPt.cx, qPt.cy, maxCanvasDist + (8 * dpr), 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(59, 130, 246, 0.12)';
                ctx.fill();
                ctx.strokeStyle = 'rgba(59, 130, 246, 0.5)';
                ctx.lineWidth = 1.5 * dpr;
                ctx.stroke();
            }
        }

        // Draw all training cluster points
        const ptRadius = 3.5 * dpr;
        const neighborRadius = 6.5 * dpr;

        mapData.points.forEach(p => {
            const pt = toCanvasCoords(p.x, p.y);
            const isNeighbor = activeNeighbors.some(n => n.index === p.index);

            ctx.beginPath();
            ctx.arc(pt.cx, pt.cy, isNeighbor ? neighborRadius : ptRadius, 0, Math.PI * 2);
            ctx.fillStyle = p.color || '#3B82F6';
            ctx.fill();

            if (isNeighbor) {
                ctx.strokeStyle = '#FFFFFF';
                ctx.lineWidth = 2.5 * dpr;
                ctx.stroke();
            }
        });

        // Draw Query Point Star / Badge
        if (query2D) {
            const qPt = toCanvasCoords(query2D[0], query2D[1]);

            // Outer glow pulse
            ctx.beginPath();
            ctx.arc(qPt.cx, qPt.cy, 14 * dpr, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(234, 179, 8, 0.3)';
            ctx.fill();

            // Inner target
            ctx.beginPath();
            ctx.arc(qPt.cx, qPt.cy, 7 * dpr, 0, Math.PI * 2);
            ctx.fillStyle = '#EAB308';
            ctx.fill();
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 2.5 * dpr;
            ctx.stroke();

            // Label
            ctx.fillStyle = '#F8FAFC';
            ctx.font = `bold ${Math.round(12 * dpr)}px Inter, sans-serif`;
            ctx.fillText('Query (Q)', qPt.cx + (10 * dpr), qPt.cy - (6 * dpr));
        }
    }

    // 3. Canvas Hover Tooltip
    canvas.addEventListener('mousemove', (e) => {
        if (!mapData || !mapData.points) return;
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const mouseCanvasX = (e.clientX - rect.left) * dpr;
        const mouseCanvasY = (e.clientY - rect.top) * dpr;
        const hitRadius = 12 * dpr;

        let found = null;
        for (let i = 0; i < mapData.points.length; i++) {
            const p = mapData.points[i];
            const pt = toCanvasCoords(p.x, p.y);
            if (Math.hypot(mouseCanvasX - pt.cx, mouseCanvasY - pt.cy) < hitRadius) {
                found = p;
                break;
            }
        }

        if (found) {
            tooltip.classList.remove('hidden');
            tooltip.style.left = `${e.clientX - rect.left + 14}px`;
            tooltip.style.top = `${e.clientY - rect.top + 14}px`;
            tooltip.textContent = `${found.name} (x: ${found.x}, y: ${found.y})`;
        } else {
            tooltip.classList.add('hidden');
        }
    });

    canvas.addEventListener('mouseleave', () => {
        if (tooltip) tooltip.classList.add('hidden');
    });

    // Helper: Base64 data URL to File object
    function dataURLtoFile(dataurl, filename) {
        const arr = dataurl.split(',');
        const mime = arr[0].match(/:(.*?);/)[1];
        const bstr = atob(arr[1]);
        let n = bstr.length;
        const u8arr = new Uint8Array(n);
        while (n--) {
            u8arr[n] = bstr.charCodeAt(n);
        }
        return new File([u8arr], filename, { type: mime });
    }

    // 4. Sample Selection (Uses instant base64 Data URL)
    async function selectSample(sample) {
        try {
            if (sample.data_url) {
                currentImageFile = dataURLtoFile(sample.data_url, sample.filename);
                currentImageSrc = sample.data_url;
            } else {
                currentImageSrc = sample.url;
                const res = await fetch(sample.url);
                const blob = await res.blob();
                currentImageFile = new File([blob], sample.filename, { type: 'image/png' });
            }
            displayPreview(currentImageSrc);
            showToast(`Loaded: ${sample.name}`);
            runPrediction();
        } catch (err) {
            console.error('Error selecting sample:', err);
            showToast('Failed to load sample image.');
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
        if (!currentImageFile && !currentImageSrc) return;
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

    // 10. Run Prediction with Direct /api/predict JSON Payload
    btnClassify.addEventListener('click', runPrediction);

    async function runPrediction() {
        if ((!currentImageFile && !currentImageSrc) || isProcessing) return;

        isProcessing = true;
        btnClassify.disabled = true;
        btnClassify.textContent = 'Classifying...';
        scanBar.classList.remove('hidden');

        const k = parseInt(kSlider.value) || 3;
        const metric = metricSelect.value;
        const weights = weightsSelect.value;

        let success = false;
        let lastErrorMessage = '';

        // Primary Strategy: JSON payload directly to /api/predict
        if (currentImageSrc) {
            try {
                const resJson = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_b64: currentImageSrc,
                        k: k,
                        metric: metric,
                        weights: weights
                    })
                });
                const data = await resJson.json();
                if (resJson.ok && data.success) {
                    renderResults(data);
                    success = true;
                } else {
                    lastErrorMessage = data.error || `Server status ${resJson.status}`;
                }
            } catch (err) {
                console.warn('JSON /api/predict failed, trying fallback:', err);
            }
        }

        // Secondary Strategy: Multipart Form Data to /api/predict
        if (!success && currentImageFile) {
            try {
                const formData = new FormData();
                formData.append('image', currentImageFile);
                formData.append('k', k);
                formData.append('metric', metric);
                formData.append('weights', weights);

                const res = await fetch('/api/predict', { method: 'POST', body: formData });
                const data = await res.json();
                if (res.ok && data.success) {
                    renderResults(data);
                    success = true;
                } else {
                    lastErrorMessage = data.error || `Server status ${res.status}`;
                }
            } catch (err) {
                console.warn('Multipart /api/predict failed:', err);
            }
        }

        // Tertiary Strategy: Fallback to root /predict
        if (!success && currentImageSrc) {
            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_b64: currentImageSrc,
                        k: k,
                        metric: metric,
                        weights: weights
                    })
                });
                const data = await res.json();
                if (res.ok && data.success) {
                    renderResults(data);
                    success = true;
                } else {
                    lastErrorMessage = data.error || `Server status ${res.status}`;
                }
            } catch (err) {
                console.warn('Fallback /predict failed:', err);
            }
        }

        if (!success) {
            showToast(lastErrorMessage ? `Error: ${lastErrorMessage}` : 'Prediction failed. Please try again.');
        }

        scanBar.classList.add('hidden');
        btnClassify.disabled = false;
        btnClassify.innerHTML = '<span>⚡ Run KNN Classification</span>';
        isProcessing = false;
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

        // Show HOG Visualizer
        if (origThumb) origThumb.src = currentImageSrc;
        if (hogThumb && data.hog_image_b64) {
            hogThumb.src = `data:image/png;base64,${data.hog_image_b64}`;
        }
        if (featureCompWrapper) featureCompWrapper.classList.remove('hidden');
        if (featureCompPlaceholder) featureCompPlaceholder.classList.add('hidden');

        // Render Voting Breakdown Bars
        renderVotingBars(uniformBars, data.voting_comparison.uniform, false);
        renderVotingBars(distanceBars, data.voting_comparison.distance, true);

        // Render Nearest Neighbors Cards
        renderNeighborsGrid(data.neighbors);
    }

    function renderVotingBars(container, list, isDistance) {
        container.innerHTML = '';
        if (!list || !list.length) {
            container.innerHTML = '<span class="text-muted">No votes</span>';
            return;
        }

        list.slice(0, 4).forEach(item => {
            const row = document.createElement('div');
            row.className = 'vote-row';

            const scoreLabel = isDistance ? `w=${item.score}` : `${item.score} votes`;
            row.innerHTML = `
                <div class="vote-row-meta">
                    <span class="truncate" title="${item.class_name}">${item.class_name}</span>
                    <span>${item.percentage}% <span style="font-size:0.6rem; color:#94A3B8;">(${scoreLabel})</span></span>
                </div>
                <div class="vote-progress-track">
                    <div class="vote-progress-fill" style="width: ${item.percentage}%; background-color: ${item.color};"></div>
                </div>
            `;
            container.appendChild(row);
        });
    }

    function renderNeighborsGrid(neighbors) {
        neighborsGrid.innerHTML = '';
        if (!neighbors || !neighbors.length) {
            neighborsGrid.innerHTML = '<div class="text-muted text-center py-2">No neighbors found.</div>';
            return;
        }

        neighbors.forEach(n => {
            const card = document.createElement('div');
            card.className = 'neighbor-card';

            const imgHtml = n.image_b64
                ? `<img src="data:image/png;base64,${n.image_b64}" class="neighbor-img" alt="${n.class_name}">`
                : `<div class="neighbor-img-placeholder" style="background-color: ${n.color};"></div>`;

            card.innerHTML = `
                <span class="neighbor-rank">#${n.rank}</span>
                ${imgHtml}
                <div class="neighbor-meta">
                    <span class="neighbor-name" title="${n.class_name}">${n.class_name}</span>
                    <span class="neighbor-dist">d: ${n.distance}</span>
                </div>
            `;
            neighborsGrid.appendChild(card);
        });
    }

    // Start App
    init();
});
