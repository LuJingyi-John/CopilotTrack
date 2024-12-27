// Point History Class: Maintains a circular buffer of point coordinates for tracking history
class PointHistory {
    constructor(maxLength) {
        this.maxLength = maxLength;
        this.data = new Float32Array(maxLength * 2); // x,y coordinates
        this.length = 0;
        this.head = 0;
    }

    // Add new point coordinates to history
    push(x, y) {
        const idx = (this.head << 1) & ((this.maxLength << 1) - 1);
        this.data[idx] = x;
        this.data[idx + 1] = y;
        this.head = (this.head + 1) & (this.maxLength - 1);
        this.length = Math.min(this.length + 1, this.maxLength);
    }

    // Get all points in history as array of {x,y} objects
    getPoints() {
        const points = [];
        let idx = ((this.head - this.length + this.maxLength) & (this.maxLength - 1)) << 1;
        for (let i = 0; i < this.length; i++) {
            points.push({
                x: this.data[idx],
                y: this.data[idx + 1]
            });
            idx = (idx + 2) & ((this.maxLength << 1) - 1);
        }
        return points;
    }
}

// Global Constants & Variables
const TRAIL_LENGTH = 40;
const COMPRESS_SIZE = 256;

// State variables
let selectedPoints = [];  // All points including hidden ones
let displayedPoints = []; // Only visible points 
let newPoints = [];      // Points pending server update
let isStreaming = false;
let animationId = null;
let isUpdating = false;
let currentLatency = 0;
let latencyUpdateInterval;
const pointsHistory = new Map();
const requestTimestamps = new Map();

// WebSocket Configuration & Event Handlers
const socket = io('http://localhost:5000', {
    reconnectionAttempts: 5,    
    reconnectionDelay: 1000,   
    timeout: 10000            
});

// Socket error handling
socket.io.on("error", (error) => {
    console.warn("Socket connection error:", error);
    alert("Connection to server failed. Please check if the server is running.");
});

socket.io.on("reconnect_attempt", (attempt) => {
    console.log(`Trying to reconnect... Attempt ${attempt}`);
});

socket.io.on("reconnect_failed", () => {
    console.warn("Failed to reconnect after all attempts");
    alert("Unable to connect to server. Please refresh the page or try again later.");
});

// Socket events
socket.on('connect', () => console.log('Connected to server'));
socket.on('disconnect', () => console.log('Disconnected from server')); 
socket.on('connect_error', error => console.error('Connection error:', error));
socket.on('reset_complete', () => {
    console.log('Tracker reset completed');
});
socket.on('reset_error', ({message}) => {
    console.error('Error resetting tracker:', message);
});
socket.on('tracking_update', ({updatedPoints, requestId}) => {
    if (requestTimestamps.has(requestId)) {
        const latency = Math.round(performance.now() - requestTimestamps.get(requestId));
        updateLatencyDisplay(latency);
        requestTimestamps.delete(requestId);
    }
    
    const scaleX = COMPRESS_SIZE / video.videoWidth;
    const scaleY = COMPRESS_SIZE / video.videoHeight;
    
    // Initialize histories for new points
    selectedPoints.forEach((point, index) => {
        if (point.isTracking && !pointsHistory.has(index)) {
            const history = new PointHistory(TRAIL_LENGTH);
            // Initialize history buffer with the point's current position
            for (let i = 0; i < TRAIL_LENGTH; i++) {
                history.push(point.x, point.y);
            }
            pointsHistory.set(index, history);
        }
    });

    // Update points and their histories using original indices
    selectedPoints = selectedPoints.map((point, index) => {
        if (updatedPoints[index] && point.isTracking) {
            const history = pointsHistory.get(index);
            if (!history) return point;

            // Get raw coordinates
            const rawX = updatedPoints[index].x / scaleX;
            const rawY = updatedPoints[index].y / scaleY;
            
            // Get current history points
            const points = history.getPoints();
            
            // Calculate smoothed coordinates
            let smoothX = rawX;
            let smoothY = rawY;
            
            if (points.length >= 3) {
                let totalWeight = 0;
                smoothX = 0;
                smoothY = 0;
                
                const numPoints = Math.min(5, points.length);
                const allPoints = [...points.slice(-numPoints + 1), {x: rawX, y: rawY}];
                
                allPoints.forEach((p, i) => {
                    const weight = i + 1;
                    smoothX += p.x * weight;
                    smoothY += p.y * weight;
                    totalWeight += weight;
                });
                
                smoothX /= totalWeight;
                smoothY /= totalWeight;
            }
            
            // Update history with smoothed coordinates
            history.push(smoothX, smoothY);
            
            return {
                ...point,
                x: smoothX,
                y: smoothY,
                isVisible: updatedPoints[index].isVisible
            };
        }
        return point;
    });
    
    updateDisplayedPoints();
    updatePointsList();
});

// DOM Elements Setup
const video = document.getElementById('videoElement');
const canvas = document.getElementById('canvasElement');
const ctx = canvas.getContext('2d');
const toggleButton = document.getElementById('toggleCamera');
const buttonText = document.getElementById('buttonText');
const pointsList = document.getElementById('pointsList');
canvas.width = 1920;
canvas.height = 1080;

// Setup compressed canvas for server updates
const compressedCanvas = document.createElement('canvas');
const compressedCtx = compressedCanvas.getContext('2d');
compressedCanvas.width = COMPRESS_SIZE;
compressedCanvas.height = COMPRESS_SIZE;

/**
 * Core Video Processing & Drawing Functions
 */
function startVideoCanvas() {
    // Server update function
    async function updatePointsFromServer() {
        if ((selectedPoints.length === 0 && !newPoints.length) || isUpdating) return;
        isUpdating = true;
        
        try {
            // Compress video frame
            compressedCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight,
                0, 0, COMPRESS_SIZE, COMPRESS_SIZE);
            
            // Prepare data for server
            const requestId = Date.now();
            const data = {
                frame: compressedCanvas.toDataURL('image/webp', 0.6),
                requestId,
                points: newPoints.length ? newPoints.map(p => ({
                    x: p.x * (COMPRESS_SIZE / video.videoWidth),
                    y: p.y * (COMPRESS_SIZE / video.videoHeight),
                    isVisible: true
                })) : undefined
            };
            
            requestTimestamps.set(requestId, performance.now());
            socket.emit('update_tracking', data);
            newPoints = [];
        } catch (err) {
            console.error("Error updating points:", err);
        } finally {
            isUpdating = false;
        }
    }

    // Draw video frame and points
    function drawFrame() {
        if (!isStreaming) return;
        
        const offscreenCanvas = new OffscreenCanvas(canvas.width, canvas.height);
        const offscreenCtx = offscreenCanvas.getContext('2d');
        
        // Calculate video dimensions
        const { offsetX, offsetY, drawWidth, drawHeight } = calculateVideoDimensions();
        
        // Draw video frame
        offscreenCtx.clearRect(0, 0, canvas.width, canvas.height);
        offscreenCtx.drawImage(video, offsetX, offsetY, drawWidth, drawHeight);
        
        // Draw points
        displayedPoints.forEach((point, i) => drawPoint(offscreenCtx, point, i));
        
        // Update main canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(offscreenCanvas, 0, 0);
        
        animationId = requestAnimationFrame(drawFrame);
    }
    
    // Helper function to calculate video dimensions
    function calculateVideoDimensions() {
        const videoAspect = video.videoWidth / video.videoHeight;
        const canvasAspect = canvas.width / canvas.height;
        let drawWidth = canvas.width;
        let drawHeight = canvas.height;
        let offsetX = 0;
        let offsetY = 0;
        
        if (videoAspect > canvasAspect) {
            drawHeight = canvas.width / videoAspect;
            offsetY = (canvas.height - drawHeight) / 2;
        } else {
            drawWidth = canvas.height * videoAspect;
            offsetX = (canvas.width - drawWidth) / 2;
        }
        
        return { offsetX, offsetY, drawWidth, drawHeight };
    }
    
    // Helper function to draw a single point
    function drawPoint(ctx, point, displayIndex) {
        // Get the original index from selectedPoints
        const originalIndex = selectedPoints.indexOf(point);
        const history = pointsHistory.get(originalIndex);
        
        // Draw history trails
        if (history?.length > 1) {
            const points = history.getPoints();
            
            // Draw trail with different style based on visibility
            if (point.isVisible) {
                // Visible points: solid blue trail
                points.forEach((p, i) => {
                    const alpha = (i / points.length) * 0.8;
                    const size = 6 * (0.3 + i/points.length * 0.7);
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, size, 0, 2 * Math.PI);
                    ctx.fillStyle = `rgba(10, 132, 255, ${alpha})`;
                    ctx.fill();
                });
            } else {
                // Invisible points: dashed red trail
                ctx.setLineDash([5, 5]);
                ctx.beginPath();
                ctx.moveTo(points[0].x, points[0].y);
                points.forEach(p => ctx.lineTo(p.x, p.y));
                ctx.strokeStyle = 'rgba(255, 59, 48, 0.5)';
                ctx.lineWidth = 2;
                ctx.stroke();
                ctx.setLineDash([]);
            }
        }
        
        // Rest of the drawing code remains the same
        ctx.beginPath();
        ctx.arc(point.x, point.y, 25 * (1 + 0.1 * Math.sin(Date.now() / 200)), 0, 2 * Math.PI);
        if (point.isVisible) {
            ctx.strokeStyle = '#30d158';
            ctx.setLineDash([]);
        } else {
            ctx.strokeStyle = '#FF3B30';
            ctx.setLineDash([5, 5]);
        }
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.setLineDash([]);
        
        ctx.beginPath();
        ctx.arc(point.x, point.y, 8, 0, 2 * Math.PI);
        if (point.isVisible) {
            ctx.fillStyle = '#0A84FF';
            ctx.shadowColor = 'rgba(10, 132, 255, 0.5)';
        } else {
            ctx.fillStyle = '#FF3B30';
            ctx.shadowColor = 'rgba(255, 59, 48, 0.5)';
        }
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.shadowBlur = 0;
        
        // Draw point number with better visibility
        const numberSize = 24;
        ctx.font = `bold ${numberSize}px SF Pro Display`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        // Draw text outline/shadow for better contrast
        const number = (displayIndex + 1).toString();
        const textY = point.y - 25;
        
        // Draw outline
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'black';
        ctx.strokeText(number, point.x, textY);
        ctx.fillStyle = 'white';
        ctx.fillText(number, point.x, textY);
    }
    
    // Start update loop
    let lastUpdateTime = 0;
    const updateLoop = async time => {
        if (!isStreaming) return;
        if (time - lastUpdateTime >= 33) {
            await updatePointsFromServer();
            lastUpdateTime = time;
        }
        requestAnimationFrame(updateLoop);
    };

    requestAnimationFrame(updateLoop);
    animationId = requestAnimationFrame(drawFrame);
}

/**
 * UI Event Handlers & Helper Functions 
 */
function updateDisplayedPoints() {
    displayedPoints = selectedPoints.filter(point => point.isTracking);
}

function updatePointsListHeight() {
    const metrics = {
        headerHeight: 30,
        pointHeight: 40,
        spacing: 6,
        padding: 28,
        minHeight: 80
    };
    
    const contentHeight = metrics.headerHeight + 
        (displayedPoints.length * metrics.pointHeight) +
        (Math.max(0, displayedPoints.length - 1) * metrics.spacing) + 
        metrics.padding;
    
    pointsList.style.height = `${Math.max(metrics.minHeight, Math.min(contentHeight, window.innerHeight - 80))}px`;
}

function updatePointsList() {
    const list = document.getElementById('pointsListItems');
    if (!list) return;
    
    // If list length doesn't match displayed points, rebuild entirely
    if (list.children.length !== displayedPoints.length) {
        list.innerHTML = ''; // Clear list
        
        // Build initial list
        selectedPoints.forEach((point, index) => {
            if (point.isTracking) {
                const pointElement = document.createElement('li');
                pointElement.className = 'point-tag';
                const displayIndex = displayedPoints.indexOf(point);
                pointElement.innerHTML = `
                    <span class="point-number">Point ${displayIndex + 1}</span>
                    <span class="tracking-status ${point.isVisible ? 'tracking-active' : 'tracking-occluded'}">
                        ${point.isVisible ? '● Tracking' : '○ Occluded'}
                    </span>
                    <button class="delete-point" onclick="deletePoint(${index})">×</button>
                `;
                list.appendChild(pointElement);
            }
        });
    } else {
        // Just update status indicators
        displayedPoints.forEach((point, i) => {
            const pointElement = list.children[i];
            if (pointElement) {
                const statusSpan = pointElement.querySelector('.tracking-status');
                if (point.isVisible !== statusSpan.classList.contains('tracking-active')) {
                    statusSpan.className = `tracking-status ${point.isVisible ? 'tracking-active' : 'tracking-occluded'}`;
                    statusSpan.textContent = point.isVisible ? '● Tracking' : '○ Occluded';
                }
            }
        });
    }
}

function deletePoint(index) {
    if (isUpdating) return;
    
    // Mark point as not tracking
    selectedPoints[index].isTracking = false;
    
    // Update displayed points
    updateDisplayedPoints();
    
    // If no points are left tracking, clear everything
    if (displayedPoints.length === 0) {
        clearAllPoints();
        return;
    }
    
    // Only remove history for the deleted point
    pointsHistory.delete(index);
    
    // Update UI
    updatePointsList();
    updatePointsListHeight();
}

function clearAllPoints() {
    if (!isUpdating) {
        selectedPoints = [];
        displayedPoints = [];
        newPoints = [];
        pointsHistory.clear();
        socket.emit('reset_tracker');
        updatePointsList();
        updatePointsListHeight();
    }
}

function updateLatencyDisplay(newLatency) {
    const latencyElement = document.querySelector('.latency-value');
    
    if (latencyUpdateInterval) clearInterval(latencyUpdateInterval);
    
    const steps = 10;
    const latencyDiff = newLatency - currentLatency;
    const stepValue = latencyDiff / steps;
    let currentStep = 0;
    
    latencyUpdateInterval = setInterval(() => {
        if (currentStep < steps) {
            currentLatency += stepValue;
            latencyElement.textContent = Math.round(currentLatency);
            currentStep++;
            
            latencyElement.classList.remove('warning', 'danger');
            if (currentLatency > 200) latencyElement.classList.add('danger');
            else if (currentLatency > 100) latencyElement.classList.add('warning');
        } else {
            clearInterval(latencyUpdateInterval);
            currentLatency = newLatency;
            latencyElement.textContent = Math.round(currentLatency);
        }
    }, 50);
}

// Event Listeners
toggleButton.addEventListener('click', async () => {
    if (!isStreaming) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {width: {ideal: 1920}, height: {ideal: 1080}}
            });
            video.srcObject = stream;
            isStreaming = true;
            toggleButton.classList.add('active');
            buttonText.textContent = 'Stop Camera';
            
            // Reset tracker before starting
            socket.emit('reset_tracker');
            
            video.onloadedmetadata = () => {
                startVideoCanvas();
            };
        } catch (err) {
            console.error("Error accessing camera:", err);
            alert("Error accessing camera. Please make sure you have granted camera permissions.");
        }
    } else {
        stopCamera();
        toggleButton.classList.remove('active');
        buttonText.textContent = 'Start Camera';
    }
});

canvas.addEventListener('click', event => {
    if (!isStreaming) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const newPoint = {
        x: (event.clientX - rect.left) * scaleX,
        y: (event.clientY - rect.top) * scaleY,
        isVisible: true,
        isTracking: true  // Deleted points will not be tracked
    };
    
    selectedPoints.push(newPoint);
    newPoints.push(newPoint);
    updateDisplayedPoints();
    updatePointsList();
    updatePointsListHeight();
});

// Cleanup Functions
function stopCamera() {
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
    video.srcObject = null;
    isStreaming = false;
    if (animationId) cancelAnimationFrame(animationId);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    socket.emit('reset_tracker');
    selectedPoints = [];
    displayedPoints = [];
    newPoints = [];
    pointsHistory.clear();
    updatePointsList();
    updatePointsListHeight();
}

// Cleanup old timestamps periodically
setInterval(() => {
    const now = Date.now();
    requestTimestamps.forEach((timestamp, requestId) => {
        if (now - timestamp > 10000) requestTimestamps.delete(requestId);
    });
}, 10000);